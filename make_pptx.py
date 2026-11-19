"""Editable slide decks. Open each .pptx in Google Slides (File → Import slides)."""

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

LABS = Path(__file__).parent
INK = "1A1A1A"
MUTED = "4A4A4A"
HEADER = "F3F3F3"


def new_deck():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    return prs


def blank(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def rgb(hex_color):
    return RGBColor.from_string(hex_color)


def bar(slide):
    shape = slide.shapes.add_shape(1, Inches(0), Inches(0), Inches(13.333), Inches(0.08))
    shape.fill.solid()
    shape.fill.fore_color.rgb = rgb("111111")
    shape.line.fill.background()


def rule(slide, top):
    shape = slide.shapes.add_shape(1, Inches(0.45), Inches(top), Inches(12.4), Inches(0.015))
    shape.fill.solid()
    shape.fill.fore_color.rgb = rgb("D0D0D0")
    shape.line.fill.background()


def add_lines(slide, left, top, width, height, lines, size=16, bold=False, color=INK, spacing=8):
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    frame = box.text_frame
    frame.word_wrap = True
    for i, line in enumerate(lines):
        paragraph = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
        paragraph.space_after = Pt(spacing)
        weight = bold
        text = line
        if isinstance(line, tuple):
            text, weight = line
        run = paragraph.add_run()
        run.text = text
        run.font.size = Pt(size)
        run.font.bold = weight
        run.font.name = "Calibri"
        run.font.color.rgb = rgb(color)
    return box


def add_table(slide, data, left, top, width, height, size=13, aligns=None):
    rows, cols = len(data), len(data[0])
    table = slide.shapes.add_table(
        rows, cols, Inches(left), Inches(top), Inches(width), Inches(height)
    ).table
    table.rows[0].height = Inches(height / rows)
    for r, row in enumerate(data):
        for c, value in enumerate(row):
            cell = table.cell(r, c)
            cell.text = "" if value is None else str(value)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            paragraph = cell.text_frame.paragraphs[0]
            align = "center" if aligns is None else aligns[c]
            paragraph.alignment = PP_ALIGN.CENTER if align == "center" else PP_ALIGN.LEFT
            if not paragraph.runs:
                paragraph.add_run()
            run = paragraph.runs[0]
            run.font.size = Pt(size)
            run.font.bold = r == 0
            run.font.name = "Calibri"
            run.font.color.rgb = rgb(INK)
            cell.fill.solid()
            cell.fill.fore_color.rgb = rgb(HEADER if r == 0 else "FFFFFF")
            cell.margin_left = Inches(0.08)
            cell.margin_right = Inches(0.08)
    return table


def title_slide(prs, title, subtitle, code_path):
    slide = blank(prs)
    bar(slide)
    add_lines(slide, 0.6, 2.15, 12, 0.8, [title], size=40, bold=True, spacing=0)
    add_lines(slide, 0.6, 3.05, 12, 0.5, [subtitle], size=20, color=MUTED, spacing=0)
    add_lines(slide, 0.6, 4.35, 12, 0.5, ["Mohamed Mahmoud"], size=22, spacing=0)
    add_lines(
        slide, 0.6, 5.15, 11, 1.3,
        [
            f"Code: {code_path}",
            "That script reproduces every number on these slides.",
            "GitHub repo:",
        ],
        size=16, color=MUTED, spacing=4,
    )


def regression(path):
    prs = new_deck()
    title_slide(
        prs,
        "Used-car sale price",
        "What the dealer should pay at auction",
        "Labs/regression/used_car_regression.py",
    )

    slide = blank(prs)
    bar(slide)
    add_lines(slide, 0.45, 0.22, 12, 0.5, ["Data and model"], size=28, bold=True, spacing=0)
    rule(slide, 0.82)
    add_lines(slide, 0.45, 1.05, 12, 0.4, ["Data"], size=18, bold=True, spacing=0)
    add_lines(
        slide, 0.45, 1.5, 12.2, 1.8,
        [
            "Source: used_car_sales.csv, with decisions on auction_lots.csv.",
            "240 closed sales. The files have no date column, so no period is stated.",
            "70/30 split, random_state 42: 168 training cars, 72 test cars.",
            "The auction file is 40 lots, scored after the model is fit.",
        ],
        size=16, spacing=6,
    )
    add_lines(slide, 0.45, 3.55, 12, 0.4, ["Model specification"], size=18, bold=True, spacing=0)
    add_lines(
        slide, 0.45, 4.05, 12.2, 2.8,
        [
            "Outcome: Sale_Price. Family: ordinary least squares.",
            "Features: Mileage_K, Age_Years, Horsepower, Accident (yes = 1).",
            "Dropped Num_Owners. Its p-value is 0.87, and training adjusted R² rises from 0.8691 to 0.8698.",
            "Training adjusted R² of the model used: 0.8698.",
            "Buy rule: predicted sale price > auction price + $1,500 prep.",
        ],
        size=16, spacing=8,
    )

    slide = blank(prs)
    bar(slide)
    add_lines(slide, 0.45, 0.22, 12, 0.5, ["Fit on the test cars, and the auction"], size=28, bold=True, spacing=0)
    rule(slide, 0.82)
    add_lines(slide, 0.4, 0.98, 6, 0.35, ["Baseline vs v1, 72 test cars"], size=16, bold=True, spacing=0)
    add_table(
        slide,
        [
            ["", "R²", "Typical error"],
            ["Baseline (train mean)", "−0.0137", "$6,494"],
            ["v1", "0.8227", "$2,716"],
        ],
        0.4, 1.45, 5.7, 1.25, size=14,
    )
    slide.shapes.add_picture(
        str(LABS / "regression" / "predicted_vs_actual.png"),
        Inches(0.45), Inches(2.95), Inches(5.5), Inches(4.2),
    )
    add_lines(slide, 6.7, 0.98, 6.2, 0.35, ["What changes for the dealer"], size=16, bold=True, spacing=0)
    add_lines(
        slide, 6.7, 1.45, 6.2, 1.4,
        ["On these 40 auction cars, buying only when the model clears the auction price plus prep turns a $6,000 loss into $15,800 of profit, worth $21,800."],
        size=16, spacing=0,
    )
    add_table(
        slide,
        [
            ["", "Cars bought", "Money-losing", "Total profit"],
            ["Baseline", "14 of 40", "10", "−$6,000"],
            ["v1", "17 of 40", "6", "$15,800"],
        ],
        6.7, 3.3, 6.1, 1.45, size=14,
    )
    prs.save(path)


def tree(path):
    prs = new_deck()
    title_slide(
        prs,
        "Clinic no-shows",
        "When to overbook a slot",
        "Labs/tree/no_show_tree.py",
    )

    slide = blank(prs)
    bar(slide)
    add_lines(slide, 0.45, 0.22, 12, 0.5, ["Data and model"], size=28, bold=True, spacing=0)
    rule(slide, 0.82)
    add_lines(slide, 0.45, 1.0, 6.2, 0.35, ["Data"], size=18, bold=True, spacing=0)
    add_lines(
        slide, 0.45, 1.45, 6.3, 1.8,
        [
            "Source: clinic_appointments.csv. 400 appointments. No date column.",
            "Stratified 70/30 split, random_state 42: 280 train, 120 test.",
            "No-show rate: 22.8% overall, 22.9% in training, 22.5% in test.",
        ],
        size=15, spacing=6,
    )
    add_lines(slide, 0.45, 3.4, 6.2, 0.35, ["Model specification"], size=18, bold=True, spacing=0)
    add_lines(
        slide, 0.45, 3.9, 6.2, 3.2,
        [
            "Outcome: no_show.",
            "Family: decision tree, depth 3, minimum leaf 15.",
            "Features: lead time, prior no-shows, age, distance, SMS reminder, new patient, appointment hour.",
            "None dropped. The fitted tree splits only on prior no-shows and SMS reminder.",
        ],
        size=15, spacing=8,
    )
    slide.shapes.add_picture(
        str(LABS / "tree" / "tree.png"),
        Inches(6.7), Inches(1.15), Inches(6.2), Inches(5.6),
    )

    slide = blank(prs)
    bar(slide)
    add_lines(slide, 0.45, 0.22, 12, 0.5, ["Test fit, and what overbooking is worth"], size=28, bold=True, spacing=0)
    rule(slide, 0.82)
    add_lines(slide, 0.45, 1.0, 12, 0.3, ["120 test appointments"], size=16, bold=True, spacing=0)
    add_lines(
        slide, 0.45, 1.38, 12, 0.35,
        ["v1 training accuracy 0.8393.     v1 test accuracy 0.8833."],
        size=15, spacing=0,
    )
    add_table(
        slide,
        [
            ["", "Hold, came", "Overbook, came", "Hold, missed", "Overbook, missed", "Accuracy", "Precision", "Recall"],
            ["Baseline, always hold", "93", "0", "27", "0", "0.7750", "0.0000", "0.0000"],
            ["v1", "86", "7", "7", "20", "0.8833", "0.7407", "0.7407"],
        ],
        0.4, 1.9, 12.5, 1.35, size=12,
    )
    add_lines(slide, 0.45, 3.55, 12, 0.35, ["What overbooking changes"], size=18, bold=True, spacing=0)
    add_lines(
        slide, 0.45, 4.05, 12.4, 0.7,
        ["Overbooking the appointments the tree marks as no-shows is worth $16.50 more per appointment."],
        size=16, bold=True, spacing=0,
    )
    add_lines(
        slide, 0.45, 4.8, 12.4, 2.3,
        [
            "Rule: overbook if prior no-shows > 2.5, or if there is no SMS reminder and prior no-shows > 0.5.",
            "Otherwise hold the slot. D (1 prior no-show, no SMS) → overbook. E (0 prior, SMS sent) → hold.",
            "Value per test appointment: baseline −$27.00, v1 −$10.50.",
            "At the file’s volume treated as one month — 400 × 12 = 4,800 appointments a year — that is $79,200 a year.",
        ],
        size=16, spacing=8,
    )
    prs.save(path)


def cluster(path):
    prs = new_deck()
    title_slide(
        prs,
        "Gym member segments",
        "Who should get the extras credit",
        "Labs/cluster/gym_segments.py",
    )

    slide = blank(prs)
    bar(slide)
    add_lines(slide, 0.45, 0.22, 12, 0.5, ["Data and model"], size=28, bold=True, spacing=0)
    rule(slide, 0.82)
    add_lines(slide, 0.45, 1.05, 12, 0.4, ["Data"], size=18, bold=True, spacing=0)
    add_lines(
        slide, 0.45, 1.55, 12.2, 1.2,
        [
            "Source: gym_members.csv. 330 members. The file has no date column, so no period is stated.",
            "70/30 split, random_state 42: 231 training members, 99 test members.",
        ],
        size=16, spacing=8,
    )
    add_lines(slide, 0.45, 2.9, 12, 0.4, ["Model specification"], size=18, bold=True, spacing=0)
    add_lines(
        slide, 0.45, 3.4, 12.2, 3.4,
        [
            "Features: age, visits per month, average visit minutes, classes per month, months as a member, monthly extras spend.",
            "Left out: member_id. It identifies the member and does not describe them.",
            "Scaling: standardize each feature on the training members, then apply that scale to test.",
            "Family: k-means. Chosen k = 4.",
            "The $10 credit goes to a segment when its center spends under $20 on extras.",
        ],
        size=16, spacing=8,
    )

    slide = blank(prs)
    bar(slide)
    add_lines(slide, 0.45, 0.22, 12, 0.45, ["Elbow, segments, and the credit"], size=28, bold=True, spacing=0)
    rule(slide, 0.78)
    slide.shapes.add_picture(
        str(LABS / "cluster" / "elbow.png"),
        Inches(0.3), Inches(0.95), Inches(6.1), Inches(3.5),
    )
    add_lines(slide, 6.7, 1.05, 6.2, 0.35, ["Why k = 4"], size=18, bold=True, spacing=0)
    add_lines(
        slide, 6.7, 1.55, 6.2, 2.4,
        [
            "Spread on the test members falls from 600.7 at k = 1 to 150.0 at k = 4 (75%), then only to 141.4, 130.6, 130.5, and 125.3.",
            "The training curve bends in the same place, so the test curve agrees with k = 4.",
        ],
        size=15, spacing=8,
    )
    add_lines(
        slide, 0.4, 4.5, 12.5, 0.35,
        ["Sending the credit to those two segments is worth $1,340 a month."],
        size=16, bold=True, spacing=0,
    )
    add_lines(
        slide, 0.4, 4.95, 12.5, 0.35,
        ["Monthly net: baseline, credit to all 330 members, −$1,200.    v1, credit to 130 members, $140."],
        size=14, spacing=0,
    )
    add_table(
        slide,
        [
            ["Segment", "Members", "Defining traits", "Extras", "Credit"],
            ["New and occasional", "70", "Age 28, 3.1 visits, member 6 months", "$9.30", "Yes"],
            ["Long-time regulars", "60", "Age 64, 12 visits, member 59 months", "$16.40", "Yes"],
            ["Solo regulars", "109", "Age 33, 16 visits, 0.9 classes", "$24.60", "No"],
            ["Class regulars", "91", "Age 42, 6.9 classes, 9.1 visits", "$46.90", "No"],
        ],
        0.35, 5.4, 9.5, 1.9, size=12,
        aligns=["left", "center", "left", "center", "center"],
    )
    add_lines(slide, 10.05, 5.4, 3.0, 0.3, ["New member"], size=14, bold=True, spacing=0)
    add_lines(
        slide, 10.05, 5.8, 3.0, 1.5,
        [
            "Age 30, 4 visits, 42 minutes, 1 class, member 5 months, extras $10.",
            "New and occasional. Gets the credit.",
        ],
        size=13, spacing=6,
    )
    prs.save(path)


if __name__ == "__main__":
    jobs = [
        (LABS / "regression" / "regression_slides.pptx", regression),
        (LABS / "tree" / "tree_slides.pptx", tree),
        (LABS / "cluster" / "cluster_slides.pptx", cluster),
    ]
    for path, build in jobs:
        build(path)
        print(path)
