import pkg_resources

libs = [
    "langchain",
    "gradio",
    "requests",
    "tqdm",
    "transformers",
    "torch",
    "pdf2image",
    "langchain_community",
]

for lib in libs:
    try:
        version = pkg_resources.get_distribution(lib).version
        print(f"{lib}: {version}")
    except pkg_resources.DistributionNotFound:
        print(f"{lib}: Not installed")