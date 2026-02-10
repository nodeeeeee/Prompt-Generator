import typer
import asyncio
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
from rich.markdown import Markdown
from typing import Optional

from src.llm_integration import LLMClient
from src.systems_engine import SystemsEngine, EngineState
from src.cloud_engine import CloudEngine, CloudState
from src.features.test_intention import IntentionTester

app = typer.Typer()
console = Console()

@app.command()
def test_intention(
    intention: str = typer.Argument(..., help="The intention to test for quality and robustness")
):
    """
    Test a software development intention for clarity, quality, and potential issues.
    """
    tester = IntentionTester()
    console.print(f"\n[bold yellow]Testing Intention:[/bold yellow] [dim]{intention[:100]}...[/dim]")
    
    with console.status("[bold green]Analyzing..."):
        result = asyncio.run(tester.test_intention(intention))
    
    if result.state.name == "COMPLETED":
        color = "green" if result.score > 70 else ("yellow" if result.score > 40 else "red")
        
        metrics_display = result.metrics.model_dump()
        
        console.print(Panel(
            f"[bold]Score:[/bold] [{color}]{result.score:.1f}/100[/{color}]\n"
            f"[bold]Valid:[/bold] {result.is_valid}\n"
            f"[bold]Metrics:[/bold] {metrics_display}\n"
            f"[bold]Feedback:[/bold]\n" + "\n".join([f" - {f}" for f in result.feedback]),
            title="Intention Test Results",
            border_style=color
        ))
    else:
        console.print(f"[bold red]Intention test failed with state {result.state.name}[/bold red]")
        if result.error_trace:
            console.print(f"[red]Error: {result.error_trace[-1].get('message')}[/red]")

async def run_generator(model: str, intention: Optional[str], domain: str = "Systems"):
    console.print(Panel(f"Welcome to the [bold blue]AI {domain} Prompt Generator[/bold blue]!", title="Welcome"))

    # 1. Setup
    try:
        client = LLMClient(default_model=model)
        if domain.lower() == "cloud":
            engine = CloudEngine(client)
            state_enum = CloudState
        else:
            engine = SystemsEngine(client)
            state_enum = EngineState
    except Exception as e:
        console.print(f"[red]Error initializing {domain} Engine: {e}[/red]")
        raise typer.Exit(code=1)

    # 2. Get Intention
    if not intention:
        intention = Prompt.ask(f"[bold green]What is your {domain.lower()} development intention?[/bold green]")

    # 3. Mode Selection
    modes = ["one-shot", "iterative", "chain-of-thought"]
    mode = Prompt.ask("\n[bold green]Select development mode[/bold green]", choices=modes, default="iterative")

    # 4. Execute Pipeline
    console.print(f"\n[yellow]Running high-performance {domain.lower()} generation pipeline...[/yellow]")
    
    with console.status("[bold green]Processing..."):
        if domain.lower() == "cloud":
             # Cloud engine can take cloud_provider, but we'll stick to basics for main CLI consistency
             context = await engine.run_pipeline(intention, model=model, mode=mode)
        else:
             context = await engine.run_pipeline(intention, model=model, mode=mode)

    # 5. Handle Results
    if context.state == state_enum.COMPLETED and context.final_prompt:
        console.print("\n" + "="*50)
        console.print(Panel(Markdown(context.final_prompt), title="Generated Prompt", subtitle="Copy this to your LLM"))
        console.print("="*50 + "\n")
        console.print(f"[dim]Generation completed in {context.metrics.total_duration_ms/1000:.2f}s[/dim]")

        # Save to file option
        save_path = Prompt.ask("[bold]Save to file? (leave empty to skip)[/bold]", default="")
        if save_path:
            try:
                with open(save_path, "w") as f:
                    f.write(context.final_prompt)
                console.print(f"[green]Saved to {save_path}[/green]")
            except Exception as e:
                console.print(f"[red]Failed to save file: {e}[/red]")
    else:
        console.print(f"\n[bold red]Pipeline failed at state {context.state.name}[/bold red]")
        for err in context.error_trace:
            msg = err.get('message', 'Unknown error')
            console.print(f"[red] - {msg}[/red]")
        
        raise typer.Exit(code=1)

@app.command()
def generate(
    model: str = typer.Option("gpt-5.2", help="Model to use (e.g., gpt-5.2, claude-4.5, gemini-3)"),
    intention: Optional[str] = typer.Option(None, help="Your initial task intention"),
    domain: str = typer.Option("Systems", help="The domain of the task (Systems or Cloud)")
):
    """
    Interactive Prompt Generator powered by Systems or Cloud Engine.
    """
    asyncio.run(run_generator(model, intention, domain))

if __name__ == "__main__":
    app()
