.PHONY: dev backend frontend build docker run install clean

install:
	cd backend && python -m pip install -r requirements.txt
	cd frontend && npm install

# Run backend + frontend in two terminals, or use:
dev:
	@echo "Run in two terminals:"
	@echo "  make backend"
	@echo "  make frontend"

backend:
	cd backend && APP_RELOAD=true python -m app.main

frontend:
	cd frontend && npm run dev

build:
	cd frontend && npm run build

docker:
	docker build -t rag-studio:0.1.0 .

run:
	docker compose up -d --build

clean:
	rm -rf frontend/dist frontend/node_modules backend/__pycache__ backend/app/__pycache__
