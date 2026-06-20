from pathlib import Path

from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject


def main() -> None:
    output = Path(__file__).resolve().parents[1] / "examples" / "docs" / "system_notes.pdf"
    output.parent.mkdir(parents=True, exist_ok=True)

    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    font = writer._add_object(
        DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
    )
    page[NameObject("/Resources")] = DictionaryObject(
        {
            NameObject("/Font"): DictionaryObject(
                {
                    NameObject("/F1"): font,
                }
            )
        }
    )
    lines = [
        "System Notes: Hybrid Retrieval",
        "First-stage retrieval favors recall and may include noisy candidates.",
        "Reranking improves precision before evidence is sent to the generator.",
        "Citation IDs must refer to chunks retrieved for the current question.",
    ]
    commands = ["BT", "/F1 12 Tf", "72 720 Td"]
    for index, line in enumerate(lines):
        escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        if index:
            commands.append("0 -24 Td")
        commands.append(f"({escaped}) Tj")
    commands.append("ET")
    stream = DecodedStreamObject()
    stream.set_data("\n".join(commands).encode("ascii"))
    page[NameObject("/Contents")] = writer._add_object(stream)
    writer.add_metadata({"/Title": "Personal RAG System Notes"})
    with output.open("wb") as target:
        writer.write(target)


if __name__ == "__main__":
    main()

