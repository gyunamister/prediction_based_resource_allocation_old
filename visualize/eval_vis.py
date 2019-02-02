import numpy as np

from bokeh.plotting import figure, show, output_file

#40, 60, 80, 100
#7567, 14917, 28393, 41371
#x = np.linspace(0.1, 5, 80)
"""
perf_x = [40, 60, 80, 100]
base_perf_y = [7567, 14917, 28393, 41371]
sug_perf_y = [7276, 14138, 24804, 37912]

p = figure(x_range=(20, 120), y_range=(5000, 45000),
           background_fill_color="#fafafa", plot_width=400, plot_height=300)
p.line(perf_x, base_perf_y, line_color='orange', legend="baseline")
p.square(perf_x, base_perf_y, fill_color=None, legend="baseline", line_color='orange')

p.line(perf_x, sug_perf_y, line_color='green', legend="suggested")
p.circle(perf_x, sug_perf_y, fill_color=None, legend="suggested", line_color="olivedrab")
p.yaxis.axis_label = "Total weighted completion time"
"""

time_x = [40, 60, 80, 100]
base_time_y = [4.153662, 4.910945, 5.640871, 6.566637]
sug_time_y = [8.443782, 11.815533, 15.113976, 18.774944]

p = figure(x_range=(20, 120), y_range=(0, 60),
           background_fill_color="#fafafa", plot_width=400, plot_height=300)
p.line(time_x, base_time_y, line_color='orange', legend="baseline")
p.square(time_x, base_time_y, fill_color=None, legend="baseline", line_color='orange')

p.line(time_x, sug_time_y, line_color='green', legend="suggested")
p.circle(time_x, sug_time_y, fill_color=None, legend="suggested", line_color="olivedrab")
p.yaxis.axis_label = "Time(secs)"

"""
p = figure(title="log axis example", y_axis_type="log",
           x_range=(0, 5), y_range=(0.001, 10**22),
           background_fill_color="#fafafa")


p.line(x, np.sqrt(x), legend="y=sqrt(x)",
       line_color="tomato", line_dash="dashed")

p.line(x, x, legend="y=x")
p.circle(x, x, legend="y=x")

p.line(x, x**2, legend="y=x**2")
p.circle(x, x**2, legend="y=x**2",
         fill_color=None, line_color="olivedrab")

p.line(x, 10**x, legend="y=10^x",
       line_color="gold", line_width=2)

p.line(x, x**x, legend="y=x^x",
       line_dash="dotted", line_color="indigo", line_width=2)

p.line(x, 10**(x**2), legend="y=10^(x^2)",
       line_color="coral", line_dash="dotdash", line_width=2)

"""
p.xaxis.axis_label = "Number of instances"
#p.xaxis.label_text_font = "times"
#p.xaxis.label_text_font_style = "italic"


#p.yaxis.label_text_font = "times"
#p.yaxis.label_text_font_style = "italic"

p.legend.location = "top_left"
p.legend.label_text_font = "times"
p.legend.label_text_font_style = "italic"
output_file("logplot.html", title="log plot example")

show(p)