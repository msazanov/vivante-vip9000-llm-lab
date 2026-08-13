#!/usr/bin/env python3
"""Create a Russian, hash-tracked E053 experiment scaffold without running a model."""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
from pathlib import Path

from tooling.experiment_validator import build_file_manifest


EXPERIMENT_ID_RE = re.compile(r"^E\d{3,}[A-Za-z0-9-]*$")


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def scaffold_experiment(parent: Path | str, experiment_id: str) -> Path:
    """Create and return a planned experiment directory."""

    if not EXPERIMENT_ID_RE.fullmatch(experiment_id):
        raise ValueError(f"некорректный идентификатор эксперимента: {experiment_id}")
    root = Path(parent).resolve() / experiment_id
    root.mkdir(parents=True, exist_ok=False)
    (root / "data").mkdir()
    (root / "results").mkdir()
    (root / "raw").mkdir()

    (root / "README.md").write_text(
        f"""# {experiment_id}: единый бенчмарк

Статус: `planned`. Модель на плате ещё не запускалась.

Цель — воспроизводимо сравнить Bonsai и LFM2.5 на Orange Pi Zero 3W с
фиксацией скорости, качества, памяти и полного трассирования. Контракт E047
закрепляет общий prompt, профили n=4 и n=32, один прогрев и пять измерений.

Перед стартом необходимо выполнить preflight всех локальных и remote-tracking
веток и сохранить его машинный манифест рядом с этой записью.
""",
        encoding="utf-8",
    )
    (root / "hypothesis-preflight.md").write_text(
        """# Предварительная проверка гипотезы

Гипотеза: единый детерминированный протокол E047 позволит сравнить скорость
Bonsai и LFM2.5 без смешения токенизаторов и форматов.

До запуска проверить все локальные и remote-tracking ветки инструментом
`tooling/branch_preflight.py`; результаты и найденные совпадения должны быть
сохранены в машинном манифесте. Скачивание моделей и инференс до завершения
проверки запрещены.
""",
        encoding="utf-8",
    )
    (root / "commands.txt").write_text(
        "# План команд; фактический инференс ещё не запускался.\n"
        "python3 tooling/branch_preflight.py --repo . --term LFM2.5 --term LFM --term E047 --term E053\n"
        "python3 tooling/experiment_validator.py --experiment . --json\n",
        encoding="utf-8",
    )
    _write_json(
        root / "environment.json",
        {
            "captured": False,
            "status": "planned",
            "note": "Заполнить версиями llama.cpp, ORT, компилятора и параметрами запуска.",
        },
    )
    _write_json(
        root / "device.json",
        {
            "captured": False,
            "status": "planned",
            "note": "Заполнить данными целевой Orange Pi A733/VIP9000 до запуска.",
        },
    )
    _write_json(
        root / "results/summary.json",
        {
            "schema_version": "e053-experiment-summary/v1",
            "experiment_id": experiment_id,
            "status": "planned",
            "generated_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "reason": "Каркас создан; измерения ещё не выполнялись.",
        },
    )

    hashed_paths = [
        "README.md",
        "hypothesis-preflight.md",
        "commands.txt",
        "environment.json",
        "device.json",
        "results/summary.json",
    ]
    _write_json(root / "data/manifest.json", build_file_manifest(root, hashed_paths, status="planned"))
    return root


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("experiments"))
    parser.add_argument("--id", dest="experiment_id", required=True)
    args = parser.parse_args()
    root = scaffold_experiment(args.root, args.experiment_id)
    print(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
