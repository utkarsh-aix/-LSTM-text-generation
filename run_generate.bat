@echo off
call .venv\Scripts\activate
python -m src.generate --seed "to be or not to be" --words 40 --temperature 0.8
pause
