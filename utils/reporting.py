from datetime import datetime
from pathlib import Path

from pyrit.executor.attack import ConsoleAttackResultPrinter
from pyrit.output.sink import FileSink


async def save_and_print_report(result, attack_type: str) -> None:
    reports_dir = Path(__file__).parent.parent / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_file_path = reports_dir / f"report_{attack_type}_{date_str}.txt"

    file_sink = FileSink(path=report_file_path)
    printer = ConsoleAttackResultPrinter(sink=file_sink, enable_colors=False)
    await printer.write_async(result=result)

    print(f"Attack outcome: {result.outcome}")
    print(f"Outcome reason: {result.outcome_reason}")
    print(f"Detailed report saved to: {report_file_path}")
