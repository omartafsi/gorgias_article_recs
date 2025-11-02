import setuptools
from setuptools import setup

TEST_DEPS = [
    "pytest==7.4.0",
    "pytest-runner==6.0.0",
    "pytest-cov==4.1.0",
    "nox==2023.4.22",
]

API_DEPS = ["pydantic==1.10.15", "fastapi==0.101.0", "uvicorn"]

GPU_DEPS = ["faiss-gpu"]

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="reranker",
    version="0.1.0",
    description="Predicting help_center articles",
    keywords=["classification", "recommendation", "embeddings"],
    author="omartafsi",
    license="MIT",
    classifiers=["Programming Language :: Python :: 3.10"],
    zip_safe=True,
    include_package_data=True,
    packages=setuptools.find_packages("src"),
    package_dir={"": "src"},
    entry_points={"console_scripts": ["article_embedding_pipeline=article_embedding_pipeline.cli.cli:main",
                                      "reranker=reranker.cli.cli:main"]},
    install_requires=requirements,
    extras_require={"test": TEST_DEPS, "api": API_DEPS, "gpu": GPU_DEPS},
)
