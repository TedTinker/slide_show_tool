#%% 
import os 

#os.chdir("C:/Users/Ted/Desktop/slide_show_tool")
print(os.getcwd())

from slide_show_tool import *



def t_name_maker(t):
    if t < 0:
        return f"m{-t}"
    else:
        return t



box_dict = {}

def make_box(
    box_type,
    t=0,
    v=0,
    text=None,
    pos=(0, 0),
    width=default_width,
    height=0.5):
    if text is None:
        if box_type == "hidden_state":
            text = fr"$d^{{\#{v}}}_{{t={t}}}$"
        elif box_type == "post":
            text = fr"$(\mu^{{q,\#{v}}}_{{t={t}}}, \sigma^{{q,\#{v}}}_{{t={t}}}) \rightarrow z^{{q,\#{v}}}_{{t={t}}}$"
        elif box_type == "prior":
            text = fr"$(\mu^{{p,\#{v}}}_{{t={t}}}, \sigma^{{p,\#{v}}}_{{t={t}}}) \rightarrow z^{{p,\#{v}}}_{{t={t}}}$"
        elif box_type == "pred_x":
            text = fr"$x^{{\#{v}}}_{{t={t}}}$"
        elif box_type == "accuracy_loss":
            text = fr"$L^{{A,\#{v}}}_{{t={t}}}$"
        elif box_type == "real_x":
            text = fr"$x_{{t={t}}}$"
        elif box_type == "complex_loss":
            text = fr"$L^{{C,\#{v}}}_{{t={t}}}$"
        else:
            text = f"Unknown box_type={box_type} (t={t}, v={v})"
    return Box(
        text=text,
        pos=pos,
        width=width,
        height=height)

            



# Step 1
input_x = Box(
    text="$x$",  
    pos=(0, 0),
    width=.1,
    height=.1)

input_x_side_text = Box(
    text="Input",  
    pos=(0, .15),
    width=0,
    height=0)

slide_1 = Slide(
    slide_title = "1: Input",
    box_list = [input_x],
    arrow_list = [],
    side_text_list = [input_x_side_text])



# Step 2
output_y = Box(
    text="$y$",  
    pos=(.25, 0),
    width=.1,
    height=.1)

output_y_side_text = Box(
    text="Output",  
    pos=(.25, .15),
    width=0,
    height=0)

input_to_output = Arrow(
    start_box = input_x, 
    stop_box = output_y, 
    start_pos = (0, 0), 
    stop_pos = (0, 0),
    text = "$w_{x, y}$")

slide_2 = Slide(
    slide_title = "2: Output",
    box_list = [input_x, output_y],
    arrow_list = [input_to_output],
    side_text_list = [input_x_side_text, output_y_side_text])



slide_list = [
    slide_1, slide_2]

min_x_pos, max_x_pos, min_y_pos, max_y_pos, center_pos = get_sizes(slide_list)

center = Box(
    text=r"",                 
    pos=center_pos,
    width=0,
    height=0)


for slide in slide_list:
    slide.plot_slide(min_x_pos, max_x_pos, min_y_pos, max_y_pos, center_pos, axes = False)
    
    
    
    
    
    
    
    
    