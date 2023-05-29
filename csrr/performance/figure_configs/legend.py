from .configs import _COLORS, _MARKERS, _LINES


def generate_legend_config(methods):
    colors = _COLORS * len(methods)
    line_styles = _LINES * len(methods)
    markers = _MARKERS * len(methods)

    legends = dict()
    for index, method in enumerate(methods):
        legend = dict(
            color=colors[index], linestyle=line_styles[index], marker=markers[index])
        legends[method] = legend

    return legends
