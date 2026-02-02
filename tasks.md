# pie chart function
def plot_piechart(data, column):
    fig, ax = plt.subplots(figsize=(7, 7))
    counts = data[column].value_counts()
    colors = plt.cm.Set2.colors

    wedges, texts, autotexts = ax.pie(
        counts,
        autopct = "%1.1f%%",
        startangle = 90,
        colors = colors,
        pctdistance = 0.75,
        wedgeprops = dict(edgecolor="white", linewidth=1)
    )

    for autotext in autotexts:
        autotext.set_color("black")
        autotext.set_fontsize(11)
        autotext.set_fontweight("bold")

    ax.legend(
        wedges,
        counts.index,
        title = f"{column} Pie Chart",
        loc = "center left",
        bbox_to_anchor = (1, 0.5),
        fontsize = 10,
        title_fontsize = 11
    )

    ax.set_title(
        f"Distribution of {column} column",
        fontsize = 14,
        fontweight = "bold",
        pad = 20
    )

    ax.axis("equal")

    plt.tight_layout()
    plt.show()
