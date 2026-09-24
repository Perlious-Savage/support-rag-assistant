run:
	python pipeline.py

run-offline:
	python pipeline.py --extractive

validate:
	python validate.py

test:
	python -m pytest -q

ask:
	python app.py --question "$(Q)"
