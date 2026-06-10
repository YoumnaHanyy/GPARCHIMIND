from service.srs_extractor import SRSExtractor

def process_srs(pdf_path: str, project_id: int, hf_key: str):
    extractor = SRSExtractor(hf_key)

    text = extractor.extract_text_from_pdf(pdf_path)

    result = extractor.extract_requirements(
        srs_text=text,
        project_id=project_id
    )

    return result