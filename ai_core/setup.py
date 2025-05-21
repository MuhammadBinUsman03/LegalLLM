from setuptools import setup, find_packages

setup(
    name="ai_core",
    version="0.1.0",
    description="AI Core package for vector database operations and embeddings",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
        "qdrant-client>=1.1.1",
        "numpy>=1.21.0",
        "transformers>=4.20.0",
        "torch>=1.10.0",
        "tqdm>=4.62.0",
        "python-multipart>=0.0.5",
    ],
    entry_points={
        "console_scripts": [
            "ai-core=ai_core.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)