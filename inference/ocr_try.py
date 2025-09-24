from docling.document_converter import DocumentConverter

source = "example_data/example.jpeg"  # file path or URL
converter = DocumentConverter()
doc = converter.convert(source).document

print(doc.export_to_markdown()) 