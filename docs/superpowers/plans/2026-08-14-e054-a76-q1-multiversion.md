# План E054: A76-multiversion ядра Q1_0 × Q8_0

> Выполнять по TDD: сначала отрицательный тест, затем минимальная реализация и повторная проверка.

**Цель:** проверить, даёт ли отдельная компиляция неизменённого 4×4 Q1-ядра под Cortex-A76 измеримый выигрыш на CPU6–7 A733 без изменения весов, порядка накопления или выходов Bonsai.

**Архитектура:** основной `repack.cpp` сохраняет stock-функцию для A55 и публичную точку входа. Новый translation unit содержит ту же арифметику, но компилируется с `-mtune=cortex-a76`; маленький диспетчер на каждом вызове выбирает A76 только для текущего CPU6/7. Режимы `stock`, `null` и `candidate` задаются до запуска процессу через `GGML_Q1_A76_DISPATCH`.

**Инструменты:** Python `unittest`, CMake/Ninja/GCC на A733, `nm`/`objdump`, существующий E004 real-tensor runner, thermal guard 85 °C.

## 1. Контракт и RED-тесты

- Добавить тесты, которые требуют: точный baseline SHA, отдельный A76 symbol/TU, `-mtune=cortex-a76`, три режима dispatch, CPU6/7-only выбор, неизменный 4×4 layout и одинаковый порядок инструкций математического тела.
- Добавить тест анализатора, который отклоняет меньше пяти пар, несовпадающие F32/Q8 SHA, non-finite timings и неверное решение при gain <12%.
- Запустить focused tests и сохранить RED.

## 2. Воспроизводимый патч и анализатор

- Сохранить единый patch к Prism llama.cpp `38c66ad0241da4f9fcce541cda8edc219086cec5`.
- Добавить анализатор сырых JSONL: пять пар по 50 измерений, pair-wise median gain, null-dispatch overhead, golden SHA и итог `PASS_FULL_GATE` либо `REJECT_MICROGATE`.
- Добавить генератор сравнительного графика только для пяти завершённых пар.
- Запустить focused и full host tests.

## 3. Изолированная сборка на Orange Pi

- Создать новый source/build каталог E054, проверить baseline hashes и применить patch; исходные каталоги не менять.
- Собрать llama.cpp и E004 runner. Зафиксировать команды, env, compiler/kernel/CPU, SHA исходников и ELF.
- Доказать через `nm`/`objdump`: оба symbols присутствуют, A76 object получил нужный флаг, вызов dispatch достижим.
- Выполнить host literal-fixture golden для `stock/null/candidate`.

## 4. Микрогейты

- Null overhead: пять чередующихся пар `stock/null`, CPU6–7, по 50 измерений; выходы F32/Q8 должны совпасть точно.
- Candidate: пять чередующихся пар `stock/candidate` на реальном `blk.0.ffn_gate.weight`, CPU6–7, по 50 измерений; точное совпадение F32/Q8 обязательно.
- Если медианный pair-wise gain A76 <12%, статус `REJECTED`, сохранить все raw и остановиться.

## 5. Полный gate только при ≥12%

- Пять пар stock/candidate, общий E047 prompt, `n=32`, deterministic decoding, настройки E035, stock DDR, thermal 85 °C.
- Требования: идентичные token IDs, candidate >1.000 tok/s и медианный gain ≥3%.
- График full строить только если этот gate реально выполнен.

## 6. Публикация

- Сохранить русскую документацию, raw stdout/stderr/JSONL/thermal, манифест SHA, patch/build/disassembly evidence и явный статус каждого провала.
- Обновить final branch preflight, запустить полный набор тестов и validator, commit/push ветки.
