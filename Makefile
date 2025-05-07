dev:
	streamlit run src/langgraphagenticai/ui/app.py

lint:
	flake8 src/

deploy:
	flyctl deploy --remote-only
