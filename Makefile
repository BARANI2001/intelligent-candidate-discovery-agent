.PHONY: install backend ui

# Install dependencies for both the API (backend) and Web (frontend)
install:
	uv sync
	cd frontend && npm install

# Run the FastAPI server
backend:
	uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Run the React client
ui:
	cd frontend && npm run dev
