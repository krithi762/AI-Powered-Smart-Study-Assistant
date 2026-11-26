import os

from app import create_app

# Determine configuration environment (defaults to 'development')
env_mode = os.getenv("FLASK_ENV", "development")
app = create_app(env_mode)

if __name__ == "__main__":
    host = os.getenv("FLASK_RUN_HOST", "127.0.0.1")
    port = int(os.getenv("FLASK_RUN_PORT", "5000"))
    debug = app.config.get("DEBUG", True)
    
    print(f" * Starting AI-Powered Smart Study Assistant on http://{host}:{port}")
    print(f" * Environment: {env_mode}")
    print(f" * Health check available at: http://{host}:{port}/health")
    
    app.run(host=host, port=port, debug=debug)
