#include <linux/module.h>
#include <linux/init.h>
#include <linux/kernel.h>
#include <linux/bio.h>
#include <linux/device-mapper.h>
#include <linux/dm-io.h>
#include <linux/slab.h>
#include <linux/vmalloc.h>
#include <linux/rcupdate.h>
#include <linux/workqueue.h>
#include <linux/bitmap.h>
#include <linux/log2.h>
#include <linux/mempool.h>

#define DM_MSG_PREFIX "virt-cow"
#define CHUNK_SIZE_SECTORS 8  /* 4KB chunks assuming 512b sectors */
#define METADATA_SIZE_SECTORS 8 /* Reserve first 4KB of CoW device for metadata */
#define SECTOR_SHIFT 9
#define SECTOR_SIZE (1 << SECTOR_SHIFT)
#define MIN_JOBS 256

/* 
 * IO Job States for the internal state machine.
 */
enum job_state {
    JOB_INITIALIZED,
    JOB_COPYING_DATA,
    JOB_UPDATING_METADATA,
    JOB_PERSISTING_METADATA,
    JOB_COMPLETING,
    JOB_ERROR
};

/*
 * Metadata structure protected by RCU.
 * Contains the bitmap tracking which chunks have been redirected to the CoW device.
 */
struct cow_metadata {
    unsigned long *valid_bitmap; /* 1 = chunk resides on CoW device, 0 = Origin */
    size_t nr_chunks;
    struct rcu_head rcu;
};

/*
 * Per-target context.
 */
struct cow_c {
    struct dm_dev *origin_dev;
    struct dm_dev *cow_dev;
    struct dm_target *ti;

    struct cow_metadata __rcu *metadata;
    
    struct workqueue_struct *wq;
    struct dm_io_client *io_client;
    
    mempool_t *job_pool;
    spinlock_t metadata_lock; /* Protects write-side metadata updates */
};

/*
 * Context for handling a specific IO request that needs CoW processing.
 */
struct cow_io_job {
    struct cow_c *context;
    struct bio *bio;
    sector_t chunk;
    struct work_struct work;
    enum job_state state;
    int error;
};

static void cow_metadata_free(struct cow_metadata *md)
{
    if (unlikely(!md))
        return;

    if (md->valid_bitmap)
        vfree(md->valid_bitmap);
    kfree(md);
}

static void cow_metadata_rcu_callback(struct rcu_head *head)
{
    struct cow_metadata *md = container_of(head, struct cow_metadata, rcu);
    cow_metadata_free(md);
}

/* 
 * Load metadata from disk.
 * Reads the bitmap from the reserved metadata area on the CoW device.
 */
static int load_metadata(struct cow_c *c, struct cow_metadata *md)
{
    struct dm_io_region io_req;
    struct dm_io_request req;
    unsigned long bitmap_bytes;
    unsigned long sectors_to_read;
    void *buffer;
    int r;

    if (unlikely(!c || !md || !md->valid_bitmap))
        return -EINVAL;

    buffer = md->valid_bitmap;
    /* 
     * Note: valid_bitmap allocation is aligned to SECTOR_SIZE in constructor 
     * to ensure we don't overflow the buffer when reading full sectors.
     */
    bitmap_bytes = BITS_TO_LONGS(md->nr_chunks) * sizeof(unsigned long);
    sectors_to_read = DIV_ROUND_UP(bitmap_bytes, SECTOR_SIZE);

    if (unlikely(sectors_to_read > METADATA_SIZE_SECTORS)) {
        DMERR("Metadata too large to load (needs %lu sectors, max %d)", sectors_to_read, METADATA_SIZE_SECTORS);
        return -E2BIG;
    }

    io_req.bdev = c->cow_dev->bdev;
    io_req.sector = 0; /* Metadata starts at sector 0 */
    io_req.count = sectors_to_read;

    req.bi_op = REQ_OP_READ;
    req.bi_op_flags = REQ_SYNC;
    req.mem.type = DM_IO_VMA;
    req.mem.ptr.vma = buffer;
    req.notify.fn = NULL;
    req.client = c->io_client;

    r = dm_io(&req, 1, &io_req, NULL);
    if (unlikely(r)) {
        DMERR("Failed to load metadata from CoW device: %d", r);
    }
    
    return r;
}

/*
 * Optimized metadata persistence.
 * Writes only the 512-byte sector containing the bit for the given chunk.
 */
static int persist_bitmap_sector(struct cow_c *c, struct cow_metadata *md, sector_t chunk)
{
    struct dm_io_region io_req;
    struct dm_io_request req;
    unsigned long *bitmap_buffer;
    unsigned long sector_index;
    void *sector_ptr;

    if (unlikely(!c || !md))
        return -EINVAL;

    bitmap_buffer = md->valid_bitmap;
    
    /* Calculate which bit, byte, and sector in the bitmap needs updating */
    /* 1 sector = 512 bytes = 4096 bits */
    sector_index = chunk / (SECTOR_SIZE * 8); 
    
    /* Pointer to the start of the sector in the vmalloc'd bitmap */
    sector_ptr = (void *)((unsigned char *)bitmap_buffer + (sector_index * SECTOR_SIZE));

    /* Safety check to ensure we don't write beyond reserved metadata area */
    if (unlikely(sector_index >= METADATA_SIZE_SECTORS)) {
        DMERR("Metadata sector index %lu out of bounds (max %d)", sector_index, METADATA_SIZE_SECTORS);
        return -E2BIG;
    }

    io_req.bdev = c->cow_dev->bdev;
    io_req.sector = sector_index; /* Offset from start of CoW device */
    io_req.count = 1; /* Write 1 sector */

    req.bi_op = REQ_OP_WRITE;
    req.bi_op_flags = REQ_SYNC | REQ_FUA; /* Ensure durability */
    req.mem.type = DM_IO_VMA;
    req.mem.ptr.vma = sector_ptr;
    req.notify.fn = NULL;
    req.client = c->io_client;

    return dm_io(&req, 1, &io_req, NULL);
}

/*
 * Perform the data copy from Origin to CoW device.
 */
static int copy_data(struct cow_c *c, sector_t chunk)
{
    struct dm_io_region src, dest;
    struct dm_io_request req;
    struct page *page;
    int r;

    if (unlikely(!c))
        return -EINVAL;

    page = alloc_page(GFP_NOIO);
    if (unlikely(!page)) {
        DMERR("Failed to allocate page for CoW (chunk %llu)", (unsigned long long)chunk);
        return -ENOMEM;
    }

    /* 1. Read data from Origin */
    src.bdev = c->origin_dev->bdev;
    src.sector = chunk * CHUNK_SIZE_SECTORS;
    src.count = CHUNK_SIZE_SECTORS;

    req.bi_op = REQ_OP_READ;
    req.bi_op_flags = REQ_SYNC;
    req.mem.type = DM_IO_PAGE;
    req.mem.ptr.p = page;
    req.mem.offset = 0;
    req.notify.fn = NULL;
    req.client = c->io_client;

    r = dm_io(&req, 1, &src, NULL);
    if (unlikely(r)) {
        DMERR("Error reading from origin device: %d (chunk %llu)", r, (unsigned long long)chunk);
        goto out_free;
    }

    /* 2. Write data to CoW Device (Data area starts after Metadata) */
    dest.bdev = c->cow_dev->bdev;
    dest.sector = METADATA_SIZE_SECTORS + (chunk * CHUNK_SIZE_SECTORS);
    dest.count = CHUNK_SIZE_SECTORS;

    req.bi_op = REQ_OP_WRITE;
    req.bi_op_flags = REQ_SYNC | REQ_FUA; /* Ensure data is safe before metadata update */
    
    r = dm_io(&req, 1, &dest, NULL);
    if (unlikely(r)) {
        DMERR("Error writing to cow device: %d (chunk %llu)", r, (unsigned long long)chunk);
    }

out_free:
    __free_page(page);
    return r;
}

/*
 * Worker function to handle Copy-on-Write logic asynchronously.
 * This function implements a state machine to manage the lifecycle of a CoW operation.
 */
static void process_cow_job(struct work_struct *work)
{
    struct cow_io_job *job = container_of(work, struct cow_io_job, work);
    struct cow_c *c = job->context;
    struct cow_metadata *md;
    int r;
    unsigned long flags;
    bool already_done = false;

    if (unlikely(!job || !c))
        return;

    /* Transition: JOB_INITIALIZED -> JOB_COPYING_DATA */
    job->state = JOB_COPYING_DATA;

    /* 
     * Race condition check: Check if another thread already completed CoW 
     * for this chunk while this job was sitting in the workqueue.
     */
    rcu_read_lock();
    md = rcu_dereference(c->metadata);
    if (test_bit(job->chunk, md->valid_bitmap)) {
        already_done = true;
    }
    rcu_read_unlock();

    if (already_done)
        goto finish_io;

    /* 
     * Step 1: Copy Data from Origin to CoW device.
     * This is the most expensive part (IO). We do this without holding any locks.
     */
    r = copy_data(c, job->chunk);
    if (unlikely(r < 0)) {
        /* Transition: -> JOB_ERROR */
        job->state = JOB_ERROR;
        job->error = r;
        bio_io_error(job->bio);
        goto out_free_job;
    }

    /* Transition: JOB_COPYING_DATA -> JOB_UPDATING_METADATA */
    job->state = JOB_UPDATING_METADATA;

    /* 
     * Step 2: Update Metadata.
     * We use a spinlock to protect the write-side of the RCU-protected bitmap.
     */
    spin_lock_irqsave(&c->metadata_lock, flags);
    
    md = rcu_dereference_protected(c->metadata, lockdep_is_held(&c->metadata_lock));
    
    /* Double-check under lock */
    if (likely(!test_bit(job->chunk, md->valid_bitmap))) {
        set_bit(job->chunk, md->valid_bitmap);
        
        /* Transition: JOB_UPDATING_METADATA -> JOB_PERSISTING_METADATA */
        job->state = JOB_PERSISTING_METADATA;

        /* 
         * Step 3: Persist Metadata to Disk.
         * We write only the affected sector of the bitmap.
         */
        r = persist_bitmap_sector(c, md, job->chunk);
        if (unlikely(r < 0)) {
             DMERR("Failed to persist metadata for chunk %llu", (unsigned long long)job->chunk);
             /* 
              * Rollback memory state: if we can't persist, we must not claim 
              * the chunk is on the CoW device.
              */
             clear_bit(job->chunk, md->valid_bitmap);
             spin_unlock_irqrestore(&c->metadata_lock, flags);
             /* Transition: -> JOB_ERROR */
             job->state = JOB_ERROR;
             job->error = r;
             bio_io_error(job->bio);
             goto out_free_job;
        }
    }
    
    spin_unlock_irqrestore(&c->metadata_lock, flags);

finish_io:
    /* Transition: -> JOB_COMPLETING */
    job->state = JOB_COMPLETING;

    /* 
     * Step 4: Finalize IO.
     * Remap the original BIO to the CoW device and resubmit.
     */
    job->bio->bi_iter.bi_sector = METADATA_SIZE_SECTORS + (job->chunk * CHUNK_SIZE_SECTORS) + 
                                  (job->bio->bi_iter.bi_sector & (CHUNK_SIZE_SECTORS - 1));
    bio_set_dev(job->bio, c->cow_dev->bdev);
    submit_bio_noacct(job->bio);

out_free_job:
    /* Return job context to the mempool */
    mempool_free(job, c->job_pool);
}

static int virt_cow_map(struct dm_target *ti, struct bio *bio)
{
    struct cow_c *c = ti->private;
    struct cow_metadata *md;
    sector_t chunk;
    struct cow_io_job *job;

    if (unlikely(!c || !bio))
        return DM_MAPIO_KILL;

    chunk = bio->bi_iter.bi_sector >> ilog2(CHUNK_SIZE_SECTORS);

    /* Boundary check */
    if (unlikely(chunk >= (ti->len >> ilog2(CHUNK_SIZE_SECTORS)))) {
        DMERR("IO sector %llu out of target bounds", (unsigned long long)bio->bi_iter.bi_sector);
        return DM_MAPIO_KILL;
    }

    rcu_read_lock();
    md = rcu_dereference(c->metadata);

    /* 
     * READ: If chunk in CoW, read CoW. Else read Origin.
     * WRITE: If chunk in CoW, write CoW. Else trigger CoW job.
     */
    
    if (test_bit(chunk, md->valid_bitmap)) {
        /* Already CoW'd, redirect to CoW device */
        bio_set_dev(bio, c->cow_dev->bdev);
        bio->bi_iter.bi_sector = METADATA_SIZE_SECTORS + (chunk * CHUNK_SIZE_SECTORS) +
                                 (bio->bi_iter.bi_sector & (CHUNK_SIZE_SECTORS - 1));
        rcu_read_unlock();
        return DM_MAPIO_REMAPPED;
    }

    if (bio_data_dir(bio) == READ) {
        /* Not CoW'd yet, read from Origin */
        bio_set_dev(bio, c->origin_dev->bdev);
        rcu_read_unlock();
        return DM_MAPIO_REMAPPED;
    }

    /* WRITE request to non-CoW chunk. Need to perform CoW. */
    
    job = mempool_alloc(c->job_pool, GFP_ATOMIC);
    if (unlikely(!job)) {
        rcu_read_unlock();
        DMERR("Failed to allocate job context from mempool");
        return DM_MAPIO_KILL;
    }

    job->context = c;
    job->bio = bio;
    job->chunk = chunk;
    job->state = JOB_INITIALIZED;
    job->error = 0;
    INIT_WORK(&job->work, process_cow_job);

    queue_work(c->wq, &job->work);
    
    rcu_read_unlock();
    return DM_MAPIO_SUBMITTED;
}

/*
 * Constructor: virt-cow <origin_dev> <cow_dev>
 */
static int virt_cow_ctr(struct dm_target *ti, unsigned int argc, char **argv)
{
    struct cow_c *c;
    struct cow_metadata *md;
    unsigned long long sectors;
    unsigned long bitmap_bytes;
    int r;

    if (unlikely(argc != 2)) {
        ti->error = "Invalid argument count";
        return -EINVAL;
    }

    c = kzalloc(sizeof(*c), GFP_KERNEL);
    if (unlikely(!c)) {
        ti->error = "Cannot allocate context";
        return -ENOMEM;
    }
    ti->private = c;
    c->ti = ti;
    spin_lock_init(&c->metadata_lock);

    /* Get Origin Device */
    r = dm_get_device(ti, argv[0], dm_table_get_mode(ti->table), &c->origin_dev);
    if (unlikely(r)) {
        ti->error = "Error opening origin device";
        goto bad_origin;
    }

    /* Get CoW Device */
    r = dm_get_device(ti, argv[1], dm_table_get_mode(ti->table), &c->cow_dev);
    if (unlikely(r)) {
        ti->error = "Error opening cow device";
        goto bad_cow;
    }

    c->wq = alloc_workqueue("virt_cow_wq", WQ_MEM_RECLAIM | WQ_UNBOUND, 0);
    if (unlikely(!c->wq)) {
        ti->error = "Cannot allocate workqueue";
        r = -ENOMEM;
        goto bad_wq;
    }

    c->io_client = dm_io_client_create();
    if (IS_ERR(c->io_client)) {
        ti->error = "Cannot create io client";
        r = PTR_ERR(c->io_client);
        goto bad_io;
    }

    c->job_pool = mempool_create_kmalloc_pool(MIN_JOBS, sizeof(struct cow_io_job));
    if (unlikely(!c->job_pool)) {
        ti->error = "Cannot create job mempool";
        r = -ENOMEM;
        goto bad_pool;
    }

    /* Initialize Metadata */
    sectors = ti->len;
    md = kzalloc(sizeof(*md), GFP_KERNEL);
    if (unlikely(!md)) {
        r = -ENOMEM;
        goto bad_md;
    }
    
    md->nr_chunks = DIV_ROUND_UP(sectors, CHUNK_SIZE_SECTORS);
    
    /* Calculate bitmap size in bytes */
    bitmap_bytes = BITS_TO_LONGS(md->nr_chunks) * sizeof(unsigned long);
    
    /* Round up to sector size to ensure we don't overflow when using dm_io/REQ_OP_READ/WRITE at sector granularity */
    bitmap_bytes = ALIGN(bitmap_bytes, SECTOR_SIZE);

    /* Ensure bitmap fits in reserved metadata area */
    if (unlikely(bitmap_bytes > (METADATA_SIZE_SECTORS * SECTOR_SIZE))) {
        ti->error = "Metadata too large for reserved area";
        kfree(md);
        r = -EINVAL;
        goto bad_md;
    }

    md->valid_bitmap = vmalloc(bitmap_bytes);
    if (unlikely(!md->valid_bitmap)) {
        kfree(md);
        r = -ENOMEM;
        goto bad_md;
    }
    
    /* Zero out the bitmap (important if load_metadata fails or doesn't read everything) */
    memset(md->valid_bitmap, 0, bitmap_bytes);

    /* Load metadata from disk */
    r = load_metadata(c, md);
    if (unlikely(r)) {
        ti->error = "Failed to load metadata";
        vfree(md->valid_bitmap);
        kfree(md);
        goto bad_md;
    }

    RCU_INIT_POINTER(c->metadata, md);

    /* Ensure bios don't span chunks to simplify logic */
    ti->max_io_len = CHUNK_SIZE_SECTORS; 

    return 0;

bad_md:
    mempool_destroy(c->job_pool);
bad_pool:
    dm_io_client_destroy(c->io_client);
bad_io:
    destroy_workqueue(c->wq);
bad_wq:
    dm_put_device(ti, c->cow_dev);
bad_cow:
    dm_put_device(ti, c->origin_dev);
bad_origin:
    kfree(c);
    return r;
}

static void virt_cow_dtr(struct dm_target *ti)
{
    struct cow_c *c = ti->private;
    struct cow_metadata *md;

    if (unlikely(!c))
        return;

    /* Flush workqueue to ensure all jobs are finished */
    if (c->wq)
        flush_workqueue(c->wq);

    /* Wait for RCU grace period before freeing metadata */
    md = rcu_dereference_protected(c->metadata, 1);
    if (md)
        call_rcu(&md->rcu, cow_metadata_rcu_callback);

    if (c->job_pool)
        mempool_destroy(c->job_pool);

    if (c->io_client)
        dm_io_client_destroy(c->io_client);

    if (c->wq)
        destroy_workqueue(c->wq);

    if (c->cow_dev)
        dm_put_device(ti, c->cow_dev);

    if (c->origin_dev)
        dm_put_device(ti, c->origin_dev);

    kfree(c);
}

static struct target_type virt_cow_target = {
    .name   = "virt-cow",
    .version = {1, 2, 0},
    .module = THIS_MODULE,
    .ctr    = virt_cow_ctr,
    .dtr    = virt_cow_dtr,
    .map    = virt_cow_map,
};

static int __init virt_cow_init(void)
{
    int r = dm_register_target(&virt_cow_target);
    if (unlikely(r < 0))
        DMERR("register failed %d", r);
    return r;
}

static void __exit virt_cow_exit(void)
{
    dm_unregister_target(&virt_cow_target);
}

module_init(virt_cow_init);
module_exit(virt_cow_exit);

MODULE_DESCRIPTION("Virt-CoW: Robust High-Performance Embedded Copy-on-Write Target");
MODULE_AUTHOR("Gemini CLI Agent - Robust Systems Engineer");
MODULE_LICENSE("GPL");
