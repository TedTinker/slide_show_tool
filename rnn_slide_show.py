#%% 
from slide_show_tool import *



default_width = .75



hidden_to_hidden = (3.5, 0)

backprop_pred_to_hidden_state_start_offset = (-.1, 0)
backprop_pred_to_hidden_state_stop_offset = (-.1, 0)

backprop_accuracy_loss_to_pred_start_offset = (-.1, 0)
backprop_accuracy_loss_to_pred_stop_offset = (-.1, 0)

backprop_prior_to_hidden_state_start_offset = (-.8, .3)
backprop_prior_to_hidden_state_stop_offset = (-.1, 0)

backprop_hidden_state_to_hidden_state_start_offset = (0, .1)
backprop_hidden_state_to_hidden_state_stop_offset = (0, .1)



def t_name_maker(t):
    if t < 0:
        return f"m{-t}"
    else:
        return t



box_dict = {}

def make_box(
    box_type,
    t=0,
    text=None,
    pos=(0, 0),
    width=default_width,
    height=0.5):
    if text is None:
        if box_type == "input_x":
            text = fr"$x_{{t={t}}}$"
        elif box_type == "hidden_state":
            text = fr"$d_{{t={t}}}$"
        elif box_type == "output_y":
            text = fr"$y_{{t={t}}}$"
        elif box_type == "accuracy_loss":
            text = fr"$L_{{t={t}}}$"
        elif box_type == "real_x":
            text = fr"$x_{{t={t}}}$"
        else:
            text = f"Unknown box_type={box_type} (t={t})"
    return Box(
        text=text,
        pos=pos,
        width=width,
        height=height)



for t in [0, 1, 2, 3]:
    t_name = t_name_maker(t)

    hidden_state_name = f"hidden_state_{t_name}"
    hidden_state_box = make_box(
        box_type = "hidden_state", t = t,
        pos = (0, 0))
    box_dict[hidden_state_name] = hidden_state_box
    
    print(box_dict.keys())
    
    input_x_name = f"input_x_{t}"
    input_x_box = make_box(
        box_type = "input_x", t = t, 
        pos = add_pos(box_dict[f"hidden_state_{t}"], (0, -1)))
    box_dict[input_x_name] = input_x_box
            
    output_y_name = f"output_y_{t}"
    output_y_box = make_box(
        box_type = "output_y", t = t, 
        pos = add_pos(box_dict[f"hidden_state_{t}"], (0, -1)))
    box_dict[output_y_name] = output_y_box
    
    accuracy_loss_name = f"accuracy_loss_{t}"
    accuracy_loss_box = make_box(
        box_type = "accuracy_loss", t = t, 
        pos = add_pos(box_dict[f"output_y_{t}"], (0, -1)))
    box_dict[accuracy_loss_name] = accuracy_loss_box
            

            
arrow_dict = {}

for t in [0, 1, 2]:
    t_name = t_name_maker(t)

    # Connect hidden_state to next hidden_state
    make_hidden_state_a_name = f"make_hidden_state_a_{t}"
    if(t == 0):
        make_hidden_state_a_arrow = Arrow(
            start_box = box_dict[f"hidden_state_{t_name_maker(t-1)}"], 
            stop_box = box_dict[f"hidden_state_{t}"],
            text = "$w_{d,d}$")
    else:
        make_hidden_state_a_arrow = Arrow(
            start_box = box_dict[f"hidden_state_{0}_{0}"], 
            stop_box = box_dict[f"hidden_state_{0}_{0}"],
            text = "$w_{d,d}$")
    arrow_dict[make_hidden_state_a_name] = make_hidden_state_a_arrow
    
    # Connect hidden to pred
    make_pred_x_0_0_name = f"make_pred_x_{t}"
    make_pred_x_arrow = Arrow(
        start_box = box_dict[f"hidden_state_{t}"], 
        stop_box = box_dict[f"pred_x_{t}"],
        text = "$w_{d,x}$")
    arrow_dict[make_pred_x_0_0_name] = make_pred_x_arrow
    
    # Connect pred to accuracy loss
    make_accuracy_loss_a_name = f"make_accuracy_loss_a_{t}"
    make_accuracy_loss_a_arrow = Arrow(
        start_box = box_dict[f"pred_x_{t}"], 
        stop_box = box_dict[f"accuracy_loss_{t}"])
    arrow_dict[make_accuracy_loss_a_name] = make_accuracy_loss_a_arrow
    
    # Connect real to accuracy loss
    make_accuracy_loss_b_name = f"make_accuracy_loss_b_{t}"
    make_accuracy_loss_b_arrow = Arrow(
        start_box = box_dict[f"real_x_{t}"], 
        stop_box = box_dict[f"accuracy_loss_{t}"])
    arrow_dict[make_accuracy_loss_b_name] = make_accuracy_loss_b_arrow
    
    # Backprop pred to hidden
    backprop_pred_hidden_state_name = f"backprop_pred_hidden_state_{t}"
    backprop_pred_hidden_state_arrow = Arrow(
        start_box = box_dict[f"pred_x_{t}"], 
        stop_box = box_dict[f"hidden_state_{t}"], 
        start_pos = backprop_pred_to_hidden_state_start_offset, 
        stop_pos = backprop_pred_to_hidden_state_stop_offset, 
        color = "red")
    arrow_dict[backprop_pred_hidden_state_name] = backprop_pred_hidden_state_arrow
    
    # Backprop accuracy_loss to pred
    backprop_accuracy_loss_pred_name = f"backprop_accuracy_loss_pred_{t}"
    backprop_accuracy_loss_pred_arrow = Arrow(
        start_box = box_dict[f"accuracy_loss_{t}"], 
        stop_box = box_dict[f"pred_x_{t}"], 
        start_pos = backprop_accuracy_loss_to_pred_start_offset, 
        stop_pos = backprop_accuracy_loss_to_pred_stop_offset, 
        color = "red")
    arrow_dict[backprop_accuracy_loss_pred_name] = backprop_accuracy_loss_pred_arrow
    
    # Backprop hidden to hidden
    backprop_hidden_state_hidden_state_name = f"backprop_hidden_state_hidden_state_{t}"
    if(t == 0):
        backprop_hidden_state_hidden_state_arrow = Arrow(
            start_box = box_dict[f"hidden_state_0"], 
            stop_box = box_dict[f"hidden_state_m1"], 
            start_pos = backprop_hidden_state_to_hidden_state_start_offset, 
            stop_pos = backprop_hidden_state_to_hidden_state_stop_offset, 
            color = "red")
    else:
        backprop_hidden_state_hidden_state_arrow = Arrow(
            start_box = box_dict[f"hidden_state_{t}"], 
            stop_box = box_dict[f"hidden_state_{t-1}"], 
            start_pos = backprop_hidden_state_to_hidden_state_start_offset, 
            stop_pos = backprop_hidden_state_to_hidden_state_stop_offset, 
            color = "red")
    arrow_dict[backprop_hidden_state_hidden_state_name] = backprop_hidden_state_hidden_state_arrow
    



# Step 1
hidden_state_m1_0_side_text = Box(
    text="Initiate hidden state/context/\nlatent state as zeroes",  
    pos=add_pos(box_dict["hidden_state_m1_0"], (2.5, -.1)),
    width=0,
    height=0)

slide_1 = Slide(
    slide_title = "1: Initiate",
    box_list = [
        box_dict["hidden_state_m1_0"]],
    arrow_list = [],
    side_text_list = [
        hidden_state_m1_0_side_text])



# Step 2
prior_0_0_side_text = Box(
    text="Generate first $t=0$\nprior inner state\n(or use $(\mu = 0, \sigma = 1)$)",  
    pos=add_pos(box_dict["prior_0_0"], (0, .75)),
    width=0,
    height=0)

slide_2 = Slide(
    slide_title = "2: Prior inner state",
    box_list = slide_1.box_list + [
        box_dict["prior_0_0"]],
    arrow_list = [
        arrow_dict["make_prior_0_0"]],
    side_text_list = [
        prior_0_0_side_text])



"""
# Step 3
hidden_state_0_0_side_text = Box(
    text="Generate first\n$t=0$ hidden state",  
    pos=add_pos(box_dict["hidden_state_0_0"], (2, 0)),
    width=0,
    height=0)

slide_3 = Slide(
    slide_title = "3: Hidden state",
    box_list = slide_2.box_list + [
        box_dict["hidden_state_0_0"]],
    arrow_list = slide_2.arrow_list + [
        arrow_dict["make_hidden_state_a_0_0"], arrow_dict["make_hidden_state_b_0_0"]],
    side_text_list = [
        hidden_state_0_0_side_text])



# Step 4
pred_x_0_0_side_text = Box(
    text="Generate first estimation of\n$t=0$ observation",  
    pos=add_pos(box_dict["pred_x_0_0"], (2.5, 0)),
    width=0,
    height=0)

slide_4 = Slide(
    slide_title = "4: Estimate observation",
    box_list = slide_3.box_list + [
        box_dict["pred_x_0_0"]],
    arrow_list = slide_3.arrow_list + [
        arrow_dict["make_pred_x_0_0"]],
    side_text_list = [
        pred_x_0_0_side_text])



# Step 5
accuracy_loss_0_0_side_text = Box(
    text="First $t=0$ Error, Loss (Accuracy)",  
    pos=add_pos(box_dict["accuracy_loss_0_0"], (3, 0)),
    width=0,
    height=0)

real_x_0_side_text = Box(
    text="Real $t=0$ observation (target)",  
    pos=add_pos(box_dict["real_x_0"], (3, 0)),
    width=0,
    height=0)

slide_5 = Slide(
    slide_title = "5: Accuracy error",
    box_list = slide_4.box_list + [
        box_dict["accuracy_loss_0_0"], 
        box_dict["real_x_0"]],
    arrow_list = slide_4.arrow_list + [
        arrow_dict["make_accuracy_loss_a_0_0"], arrow_dict["make_accuracy_loss_b_0_0"]],
    side_text_list = [
        accuracy_loss_0_0_side_text, 
        real_x_0_side_text])



# Step 6
complex_loss_0_0_side_text = Box(
    text="First $t=0$ prior/posterior\nDKL, Loss (Complexity)",  
    pos=add_pos(box_dict["hidden_state_0_0"], (3.25, 2)),
    width=0,
    height=0)

slide_6 = Slide(
    slide_title = "6: Complexity error",
    box_list = slide_5.box_list + [
        box_dict["complex_loss_0_0"]],
    arrow_list = slide_5.arrow_list + [
        arrow_dict["make_complex_loss_a_0_0"], arrow_dict["make_complex_loss_b_0_0"]],
    side_text_list = [
        complex_loss_0_0_side_text])



# Step 7
first_backprop_side_text = Box(
    text="With backpropagation,\nupdate hidden state $t=-1$ and\nposterior inner state for t = 0",  
    pos=add_pos(box_dict["hidden_state_0_0"], (3, 0)),
    width=0,
    height=0)

slide_7 = Slide(
    slide_title = "7: Update",
    box_list = slide_6.box_list + [],
    arrow_list = slide_6.arrow_list + [
        arrow_dict["backprop_hidden_state_post_0_0"],
        arrow_dict["backprop_complex_loss_post_0_0"],
        arrow_dict["backprop_complex_loss_prior_0_0"],
        arrow_dict["backprop_pred_hidden_state_0_0"],
        arrow_dict["backprop_accuracy_loss_pred_0_0"],
        arrow_dict["backprop_prior_hidden_state_0_0"],
        arrow_dict["backprop_hidden_state_hidden_state_0_0"]],
    side_text_list = [
        first_backprop_side_text])



# Step 8
first_backprop_complete_side_text = Box(
    text="Values updated",  
    pos=(.25, 1),
    width=0,
    height=0)

slide_8 = Slide(
    slide_title = "8: Update",
    box_list = [
        box_dict["hidden_state_m1_1"],
        box_dict["post_0_1"]],
    arrow_list = [],
    side_text_list = [
        first_backprop_complete_side_text])



# Step 9
do_that_again_side_text = Box(
    text="Perform the previous steps\nfor the second time",  
    pos=(6, 0),
    width=0,
    height=0)

slide_9 = Slide(
    slide_title = "9: Do this again",
    box_list = slide_8.box_list + [
        box_dict["prior_0_1"],
        box_dict["hidden_state_0_1"], 
        box_dict["pred_x_0_1"], 
        box_dict["accuracy_loss_0_1"],
        box_dict["real_x_0"],
        box_dict["complex_loss_0_1"]],
    arrow_list = [
        arrow_dict["make_prior_0_1"],
        arrow_dict["make_hidden_state_a_0_1"], arrow_dict["make_hidden_state_b_0_1"],
        arrow_dict["make_pred_x_0_1"],
        arrow_dict["make_accuracy_loss_a_0_1"], arrow_dict["make_accuracy_loss_b_0_1"],
        arrow_dict["make_complex_loss_a_0_1"], arrow_dict["make_complex_loss_b_0_1"]],
    side_text_list = [
        do_that_again_side_text])



# Step 10
post_1_0_side_text = Box(
    text="Initiate first $t=1$ posterior\ninner state $(\mu = 0, \sigma = 1)$",  
    pos=add_pos(box_dict["post_1_0"], (0, -1)),
    width=0,
    height=0)

slide_10 = Slide(
    slide_title = "10: Initiate further",
    box_list = slide_9.box_list + [
        box_dict["post_1_0"]],
    arrow_list = slide_9.arrow_list + [],
    side_text_list = [
        post_1_0_side_text])



# Step 11
hidden_state_1_0_side_text = Box(
    text="Generate first\n$t=1$ hidden state",  
    pos=add_pos(box_dict["hidden_state_1_0"], (0, -1)),
    width=0,
    height=0)

make_hidden_state_1_0_a = Arrow(
    start_box=box_dict["hidden_state_0_1"], 
    stop_box=box_dict["hidden_state_1_0"])

make_hidden_state_1_0_b = Arrow(
    start_box=box_dict["post_1_0"], 
    stop_box=box_dict["hidden_state_1_0"])

slide_11 = Slide(
    slide_title = "11: Continue further",
    box_list = slide_10.box_list + [
        box_dict["hidden_state_1_0"]],
    arrow_list = slide_10.arrow_list + [
        arrow_dict["make_hidden_state_a_1_1"], arrow_dict["make_hidden_state_b_1_0"]],
    side_text_list = [
        hidden_state_1_0_side_text])



# Step 12
do_that_again_side_text = Box(
    text="Perform the previous\nsteps for $t=1$",  
    pos=(1, 2.5),
    width=0,
    height=0)

slide_12 = Slide(
    slide_title = "12: Do this again",
    box_list = slide_11.box_list + [
        box_dict["prior_1_0"],
        box_dict["pred_x_1_0"], 
        box_dict["accuracy_loss_1_0"],
        box_dict["real_x_1"],
        box_dict["complex_loss_1_0"],
        box_dict["post_0_1"], 
        box_dict["hidden_state_0_1"]],
    arrow_list = slide_11.arrow_list + [
        arrow_dict["make_prior_1_1"],
        arrow_dict["make_pred_x_1_0"],
        arrow_dict["make_accuracy_loss_a_1_0"], arrow_dict["make_accuracy_loss_b_1_0"],
        arrow_dict["make_complex_loss_a_1_0"], arrow_dict["make_complex_loss_b_1_0"]],
    side_text_list = [
        do_that_again_side_text])



# Step 13
second_backprop_side_text = Box(
    text="With backpropagation,\nupdate hidden state $t=-1$ and\nposterior inner state for t = 0, 1",  
    pos=(1.5, 2.5),
    width=0,
    height=0)

slide_13 = Slide(
    slide_title = "13: Update",
    box_list = slide_12.box_list + [],
    arrow_list = slide_12.arrow_list + [
        arrow_dict["backprop_hidden_state_post_0_1"],
        arrow_dict["backprop_hidden_state_post_1_0"],
        arrow_dict["backprop_complex_loss_post_0_1"],
        arrow_dict["backprop_complex_loss_post_1_0"],
        arrow_dict["backprop_complex_loss_prior_0_1"],
        arrow_dict["backprop_complex_loss_prior_1_0"],
        arrow_dict["backprop_pred_hidden_state_0_1"],
        arrow_dict["backprop_pred_hidden_state_1_0"],
        arrow_dict["backprop_accuracy_loss_pred_0_1"],
        arrow_dict["backprop_accuracy_loss_pred_1_0"],
        arrow_dict["backprop_prior_hidden_state_0_1"],
        arrow_dict["backprop_prior_hidden_state_1_0"],
        arrow_dict["backprop_hidden_state_hidden_state_0_1"],
        arrow_dict["backprop_hidden_state_hidden_state_1_0"]],
    side_text_list = [
        second_backprop_side_text,])



# Step 14
second_backprop_complete_side_text = Box(
    text="Values updated",  
    pos=(.25, 1),
    width=0,
    height=0)

slide_14 = Slide(
    slide_title = "14: Update",
    box_list = [
        box_dict["hidden_state_m1_2"],
        box_dict["post_0_2"],
        box_dict["post_1_1"]],
    arrow_list = [],
    side_text_list = [
        second_backprop_complete_side_text])

"""

slide_list = [
    slide_1]

min_x_pos, max_x_pos, min_y_pos, max_y_pos, center_pos = get_sizes(slide_list)

center = Box(
    text=r"",                 
    pos=center_pos,
    width=0,
    height=0)



for slide in slide_list:
    slide.plot_slide(min_x_pos, max_x_pos, min_y_pos, max_y_pos, center_pos, axes = False)
    
    
    
    
    