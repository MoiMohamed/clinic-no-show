"""Build the three lab slide PDFs from the numbers the analysis scripts print.

Each deck is a title slide plus two content slides. Re-run the analysis
scripts first if the data or the model code changes, then update the
figures below so the slides still match the code.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
LABS = Path(__file__).parent
W, H = 13.333, 7.5
INK = "#1a1a1a"
MUTED = "#4a4a4a"
RULE = "#d0d0d0"
HEAD = "#111111"


def new_slide(title=None):
    fig = plt.figure(figsize=(W, H), dpi=120)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0, H - 0.08), W, 0.08, color=HEAD, lw=0))
    if title:
        ax.text(0.48, H - 0.38, title, fontsize=20, fontweight="bold", color=INK, va="top")
        ax.plot([0.48, W - 0.48], [H - 0.72, H - 0.72], color=RULE, lw=1)
    return fig, ax


def plain(s):
    return str(s).replace("$", r"\$")


def text(ax, x, y, s, size=12, weight="normal", color=INK, ha="left", va="top"):
    return ax.text(
        x, y, plain(s), fontsize=size, fontweight=weight, color=color, ha=ha, va=va,
        linespacing=1.35,
    )


def place_image(fig, path, left, bottom, width, height):
    ax = fig.add_axes([left, bottom, width, height])
    ax.imshow(plt.imread(path))
    ax.axis("off")
    return ax


def bullets(ax, x, y, lines, size=12.5, leading=0.34, color=INK):
    for i, line in enumerate(lines):
        ax.text(x, y - i * leading, plain(line), fontsize=size, color=color, va="top")


def fill_table(ax, rows, bbox, size=11, col_widths=None):
    table = ax.table(
        cellText=[[plain(c) for c in row] for row in rows],
        cellLoc="center",
        bbox=bbox,
        colWidths=col_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(size)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor(RULE)
        cell.set_linewidth(0.6)
        cell.set_facecolor("#f3f3f3" if r == 0 else "white")
        if r == 0:
            cell.set_text_props(fontweight="bold")
    return table


def save(pdf, fig):
    pdf.savefig(fig)
    plt.close(fig)


def regression():
    out = LABS / "regression" / "regression_slides.pdf"
    chart = LABS / "regression" / "predicted_vs_actual.png"
    with PdfPages(out) as pdf:
        fig, ax = new_slide()
        text(ax, 0.55, 5.7, "Used-car sale price", size=32, weight="bold")
        text(ax, 0.55, 4.85, "What the dealer should pay at auction", size=16, color=MUTED)
        text(ax, 0.55, 3.3, "Mohamed Mahmoud", size=18)
        text(
            ax, 0.55, 2.55,
            "Code: Labs/regression/used_car_regression.py\n"
            "That script reproduces every number on these slides.\n"
            "GitHub repo: not published — Labs is not a git repository.",
            size=13, color=MUTED,
        )
        save(pdf, fig)

        fig, ax = new_slide("Data and model")
        text(ax, 0.48, 6.45, "Data", size=15, weight="bold")
        bullets(ax, 0.48, 6.05, [
            "Source: used_car_sales.csv, with decisions on auction_lots.csv.",
            "240 closed sales. The files have no date column, so no period is stated.",
            "70/30 split, random_state 42: 168 training cars, 72 test cars.",
            "The auction file is 40 lots, scored after the model is fit.",
        ], size=12.5, leading=0.38)

        text(ax, 0.48, 4.25, "Model specification", size=15, weight="bold")
        bullets(ax, 0.48, 3.85, [
            "Outcome: Sale_Price. Family: ordinary least squares.",
            "Features: Mileage_K, Age_Years, Horsepower, Accident (yes = 1).",
            "Dropped Num_Owners. Its p-value is 0.87, and training adjusted R²",
            "    rises from 0.8691 to 0.8698 without it.",
            "Training adjusted R² of the model used: 0.8698.",
            "Buy rule: predicted sale price > auction price + $1,500 prep.",
        ], size=12.5, leading=0.36)
        save(pdf, fig)

        fig, ax = new_slide("Fit on the test cars, and the auction")
        text(ax, 0.48, 6.45, "Baseline vs v1, 72 test cars", size=14, weight="bold")
        fill_table(
            ax,
            [
                ["", "R²", "Typical error"],
                ["Baseline (train mean)", "−0.0137", "$6,494"],
                ["v1", "0.8227", "$2,716"],
            ],
            [0.035, 0.60, 0.42, 0.16],
            size=11,
        )
        place_image(fig, chart, 0.045, 0.08, 0.38, 0.48)

        text(ax, 7.15, 6.45, "What changes for the dealer", size=14, weight="bold")
        text(
            ax, 7.15, 5.95,
            "On these 40 auction cars, buying only when the\n"
            "model clears the auction price plus prep turns\n"
            "a $6,000 loss into $15,800 of profit, worth $21,800.",
            size=12, color=INK,
        )
        fill_table(
            ax,
            [
                ["", "Cars bought", "Money-losing", "Total profit"],
                ["Baseline", "14 of 40", "10", "−$6,000"],
                ["v1", "17 of 40", "6", "$15,800"],
            ],
            [0.52, 0.22, 0.44, 0.26],
            size=11,
        )
        save(pdf, fig)
    return out


def tree():
    out = LABS / "tree" / "tree_slides.pdf"
    picture = LABS / "tree" / "tree.png"
    with PdfPages(out) as pdf:
        fig, ax = new_slide()
        text(ax, 0.55, 5.7, "Clinic no-shows", size=32, weight="bold")
        text(ax, 0.55, 4.85, "When to overbook a slot", size=16, color=MUTED)
        text(ax, 0.55, 3.3, "Mohamed Mahmoud", size=18)
        text(
            ax, 0.55, 2.55,
            "Code: Labs/tree/no_show_tree.py\n"
            "That script reproduces every number on these slides.\n"
            "GitHub repo: not published — Labs is not a git repository.",
            size=13, color=MUTED,
        )
        save(pdf, fig)

        fig, ax = new_slide("Data and model")
        text(ax, 0.48, 6.45, "Data", size=15, weight="bold")
        bullets(ax, 0.48, 6.05, [
            "Source: clinic_appointments.csv. 400 appointments. No date column.",
            "Stratified 70/30 split, random_state 42: 280 train, 120 test.",
            "No-show rate: 22.8% overall, 22.9% in training, 22.5% in test.",
        ], size=12.5, leading=0.36)
        text(ax, 0.48, 4.75, "Model specification", size=15, weight="bold")
        bullets(ax, 0.48, 4.35, [
            "Outcome: no_show.",
            "Family: decision tree, depth 3,",
            "minimum leaf 15.",
            "Features: lead time, prior no-shows,",
            "age, distance, SMS reminder,",
            "new patient, appointment hour.",
            "None dropped. Splits used:",
            "prior no-shows and SMS reminder.",
        ], size=12.5, leading=0.34)
        place_image(fig, picture, 0.50, 0.06, 0.48, 0.55)
        save(pdf, fig)

        fig, ax = new_slide("Test fit, and what overbooking is worth")
        text(ax, 0.48, 6.48, "120 test appointments", size=14, weight="bold")
        text(
            ax, 0.48, 6.08,
            "v1 training accuracy 0.8393.    v1 test accuracy 0.8833.",
            size=12,
        )
        fill_table(
            ax,
            [
                ["", "Hold, came", "Overbook, came", "Hold, missed", "Overbook, missed", "Accuracy", "Precision", "Recall"],
                ["Baseline, always hold", "93", "0", "27", "0", "0.7750", "0.0000", "0.0000"],
                ["v1", "86", "7", "7", "20", "0.8833", "0.7407", "0.7407"],
            ],
            [0.04, 0.60, 0.92, 0.15],
            size=9,
        )

        text(ax, 0.48, 4.15, "What overbooking changes", size=14, weight="bold")
        text(
            ax, 0.48, 3.75,
            "Overbooking the appointments the tree marks as no-shows is worth $16.50 more per appointment.",
            size=12.5,
        )
        bullets(ax, 0.48, 3.30, [
            "Rule: overbook if prior no-shows > 2.5, or if there is no SMS reminder and prior no-shows > 0.5.",
            "Otherwise hold the slot.  D (1 prior no-show, no SMS) → overbook.  E (0 prior, SMS sent) → hold.",
            "Value per test appointment: baseline −$27.00, v1 −$10.50.",
            "At the file’s volume treated as one month — 400 × 12 = 4,800 appointments a year — that is $79,200 a year.",
        ], size=12, leading=0.38)
        save(pdf, fig)
    return out


def cluster():
    out = LABS / "cluster" / "cluster_slides.pdf"
    chart = LABS / "cluster" / "elbow.png"
    with PdfPages(out) as pdf:
        fig, ax = new_slide()
        text(ax, 0.55, 5.7, "Gym member segments", size=32, weight="bold")
        text(ax, 0.55, 4.85, "Who should get the extras credit", size=16, color=MUTED)
        text(ax, 0.55, 3.3, "Mohamed Mahmoud", size=18)
        text(
            ax, 0.55, 2.55,
            "Code: Labs/cluster/gym_segments.py\n"
            "That script reproduces every number on these slides.\n"
            "GitHub repo: not published — Labs is not a git repository.",
            size=13, color=MUTED,
        )
        save(pdf, fig)

        fig, ax = new_slide("Data and model")
        text(ax, 0.48, 6.45, "Data", size=15, weight="bold")
        bullets(ax, 0.48, 6.05, [
            "Source: gym_members.csv. 330 members. The file has no date column, so no period is stated.",
            "70/30 split, random_state 42: 231 training members, 99 test members.",
        ], size=12.5, leading=0.38)
        text(ax, 0.48, 5.05, "Model specification", size=15, weight="bold")
        bullets(ax, 0.48, 4.65, [
            "Features: age, visits per month, average visit minutes, classes per month,",
            "    months as a member, monthly extras spend.",
            "Left out: member_id. It identifies the member and does not describe them.",
            "Scaling: standardize each feature on the training members, then apply that scale to test.",
            "Family: k-means. Chosen k = 4.",
            "The $10 credit goes to a segment when its center spends under $20 on extras.",
        ], size=12.5, leading=0.36)
        save(pdf, fig)

        fig, ax = new_slide("Elbow, segments, and the credit")
        place_image(fig, chart, 0.03, 0.50, 0.40, 0.38)
        text(ax, 6.9, 6.5, "Why k = 4", size=14, weight="bold")
        text(
            ax, 6.9, 6.05,
            "Spread on the test members falls from\n"
            "600.7 at k = 1 to 150.0 at k = 4 (75%),\n"
            "then only to 141.4, 130.6, 130.5, 125.3.\n"
            "The training curve bends in the same place,\n"
            "so the test curve agrees with k = 4.",
            size=12,
        )
        text(
            ax, 0.48, 3.50,
            "Sending the credit to those two segments is worth $1,340 a month.",
            size=13, weight="bold",
        )
        text(
            ax, 0.48, 3.12,
            "Monthly net: baseline, credit to all 330 members, −$1,200.    v1, credit to 130 members, $140.",
            size=12,
        )
        fill_table(
            ax,
            [
                ["Segment", "Members", "Defining traits", "Extras", "Credit"],
                ["New and occasional", "70", "Age 28, 3.1 visits, member 6 months", "$9.30", "Yes"],
                ["Long-time regulars", "60", "Age 64, 12 visits, member 59 months", "$16.40", "Yes"],
                ["Solo regulars", "109", "Age 33, 16 visits, 0.9 classes", "$24.60", "No"],
                ["Class regulars", "91", "Age 42, 6.9 classes, 9.1 visits", "$46.90", "No"],
            ],
            [0.03, 0.05, 0.68, 0.28],
            size=9,
            col_widths=[0.24, 0.12, 0.40, 0.12, 0.12],
        )
        text(ax, 9.7, 2.40, "New member", size=12, weight="bold")
        text(
            ax, 9.7, 2.05,
            "Age 30, 4 visits,\n"
            "42 minutes, 1 class,\n"
            "member 5 months,\n"
            "extras $10.\n\n"
            "New and occasional.\n"
            "Gets the credit.",
            size=11,
        )
        save(pdf, fig)
    return out


if __name__ == "__main__":
    for path in (regression(), tree(), cluster()):
        print(path)
