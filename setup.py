from setuptools import setup, find_packages

with open('requirements.txt', "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="skku-skku/summarize-model",
    version="0.0.1",
    author="yejin",
    author_email="ssgyejin@gmail.com",
    description="kobart summarization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/skku-skku/summarize-model.git",
    packages=find_packages(),
    install_requires=requirements,
)