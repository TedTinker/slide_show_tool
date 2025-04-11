#%% 
from slide_show_tool import *



default_width = .75
prior_post_width = 2.5



hidden_to_hidden = (3.5, 0)

hidden_to_prior = (2.75, 1.6)
hidden_to_prior_stop_offset = (-.8, .2)

hidden_to_post = (2.75, 1)
hidden_to_post_start_offset = (.6, 0)
hidden_to_post_stop_offset = (.2, 0)

post_to_complex = (1.75, 1)
post_to_complex_start_offset = (.8, -.2)
post_to_complex_stop_offset = (.5, 0)

prior_to_complex_start_offset = (.4, 0)
prior_to_complex_stop_offset = (0, 0)

backprop_hidden_to_post_start_offset = (.1, 0)
backprop_hidden_to_post_stop_offset = (.5, 0)

backprop_complex_loss_to_post_start_offset = (.4, 0)
backprop_complex_loss_to_post_stop_offset = (.8, -.1)

backprop_complex_loss_to_prior_start_offset = (0, .1)
backprop_complex_loss_to_prior_stop_offset = (.3, 0)

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



for t in [-1, 0, 1, 2, 3]:
    t_name = t_name_maker(t)
    for v in [0, 1, 2, 3, 4]:
        hidden_state_name = f"hidden_state_{t_name}_{v}"
        hidden_state_box = make_box(
            box_type = "hidden_state", t = t, v = v, 
            pos = (0, 0) if t == -1 else add_pos(box_dict[f"hidden_state_{t_name_maker(t-1)}_{v}"], hidden_to_hidden))
        box_dict[hidden_state_name] = hidden_state_box
        
        if(t >= 0):
            prior_name = f"prior_{t}_{v}"
            prior_box = make_box(
                box_type = "prior", t = t, v = v, 
                pos = add_pos(box_dict[f"hidden_state_{t_name_maker(t-1)}_{v}"], hidden_to_prior), width = prior_post_width)
            box_dict[prior_name] = prior_box
            
            post_name = f"post_{t}_{v}"
            post_box = make_box(
                box_type = "post", t = t, v = v, 
                pos = add_pos(box_dict[f"hidden_state_{t_name_maker(t-1)}_{v}"], hidden_to_post), width = prior_post_width)
            box_dict[post_name] = post_box
            
            pred_x_name = f"pred_x_{t}_{v}"
            pred_x_box = make_box(
                box_type = "pred_x", t = t, v = v, 
                pos = add_pos(box_dict[f"hidden_state_{t}_{v}"], (0, -1)))
            box_dict[pred_x_name] = pred_x_box
            
            accuracy_loss_name = f"accuracy_loss_{t}_{v}"
            accuracy_loss_box = make_box(
                box_type = "accuracy_loss", t = t, v = v,
                pos = add_pos(box_dict[f"pred_x_{t}_{v}"], (0, -1)))
            box_dict[accuracy_loss_name] = accuracy_loss_box
            
            complex_loss_name = f"complex_loss_{t}_{v}"
            complex_box = make_box(
                box_type = "complex_loss", t = t, v = v,
                pos = add_pos(box_dict[f"post_{t}_{v}"], post_to_complex))
            box_dict[complex_loss_name] = complex_box
            
    if(t >= 0):
        real_x_name = f"real_x_{t}"
        real_x_box = make_box(
            box_type = "real_x", t = t, v = None,
            pos = add_pos(box_dict[f"accuracy_loss_{t}_0"], (0, -1)))
        box_dict[real_x_name] = real_x_box
            

            
arrow_dict = {}

for t in [0, 1, 2]:
    t_name = t_name_maker(t)
    for v in [0, 1, 2, 3]:
        # Connect hidden state to next prior
        make_prior_name = f"make_prior_{t}_{v}"
        if(t == 0):
            make_prior_arrow = Arrow(
                start_box = box_dict[f"hidden_state_{t_name_maker(t-1)}_{v}"], 
                stop_box = box_dict[f"prior_{t}_{v}"], 
                stop_pos = hidden_to_prior_stop_offset,
                text = "$w_{d,p}$")
        elif(v > 0):
            make_prior_arrow = Arrow(
                start_box = box_dict[f"hidden_state_{t_name_maker(t-1)}_{v}"], 
                stop_box = box_dict[f"prior_{t}_{v-1}"], 
                stop_pos = hidden_to_prior_stop_offset,
                text = "$w_{d,p}$")
        else:
            make_prior_arrow = Arrow(
                start_box = box_dict[f"hidden_state_{0}_{0}"], 
                stop_box = box_dict[f"prior_{0}_{0}"], 
                stop_pos = hidden_to_prior_stop_offset,
                text = "$w_{d,p}$")
        arrow_dict[make_prior_name] = make_prior_arrow
        
        # Connect hidden_state to next hidden_state
        make_hidden_state_a_name = f"make_hidden_state_a_{t}_{v}"
        if(t == 0):
            make_hidden_state_a_arrow = Arrow(
                start_box = box_dict[f"hidden_state_{t_name_maker(t-1)}_{v}"], 
                stop_box = box_dict[f"hidden_state_{t}_{v}"],
                text = "$w_{d,d}$")
        elif(v > 0):
            make_hidden_state_a_arrow = Arrow(
                start_box = box_dict[f"hidden_state_{t_name_maker(t-1)}_{v}"], 
                stop_box = box_dict[f"hidden_state_{t}_{v-1}"],
                text = "$w_{d,d}$")
        else:
            make_hidden_state_a_arrow = Arrow(
                start_box = box_dict[f"hidden_state_{0}_{0}"], 
                stop_box = box_dict[f"hidden_state_{0}_{0}"],
                text = "$w_{d,d}$")
        arrow_dict[make_hidden_state_a_name] = make_hidden_state_a_arrow
        
        # Connect post to hidden
        make_hidden_state_b_name = f"make_hidden_state_b_{t}_{v}"
        make_hidden_state_b_arrow = Arrow(
            start_box = box_dict[f"post_{t}_{v}"], 
            stop_box = box_dict[f"hidden_state_{t}_{v}"], 
            start_pos = hidden_to_post_start_offset, 
            stop_pos = hidden_to_post_stop_offset,
            text = "$w_{q,d}$")
        arrow_dict[make_hidden_state_b_name] = make_hidden_state_b_arrow
        
        # Connect hidden to pred
        make_pred_x_0_0_name = f"make_pred_x_{t}_{v}"
        make_pred_x_arrow = Arrow(
            start_box = box_dict[f"hidden_state_{t}_{v}"], 
            stop_box = box_dict[f"pred_x_{t}_{v}"],
            text = "$w_{d,x}$")
        arrow_dict[make_pred_x_0_0_name] = make_pred_x_arrow
        
        # Connect pred to accuracy loss
        make_accuracy_loss_a_name = f"make_accuracy_loss_a_{t}_{v}"
        make_accuracy_loss_a_arrow = Arrow(
            start_box = box_dict[f"pred_x_{t}_{v}"], 
            stop_box = box_dict[f"accuracy_loss_{t}_{v}"])
        arrow_dict[make_accuracy_loss_a_name] = make_accuracy_loss_a_arrow
        
        # Connect real to accuracy loss
        make_accuracy_loss_b_name = f"make_accuracy_loss_b_{t}_{v}"
        make_accuracy_loss_b_arrow = Arrow(
            start_box = box_dict[f"real_x_{t}"], 
            stop_box = box_dict[f"accuracy_loss_{t}_{v}"])
        arrow_dict[make_accuracy_loss_b_name] = make_accuracy_loss_b_arrow
        
        # Connect prior to complex loss
        make_complex_loss_a_name = f"make_complex_loss_a_{t}_{v}"
        make_complex_loss_a_arrow = Arrow(
            start_box = box_dict[f"prior_{t}_{v}"], 
            stop_box = box_dict[f"complex_loss_{t}_{v}"], 
            start_pos = prior_to_complex_start_offset, 
            stop_pos = prior_to_complex_stop_offset)
        arrow_dict[make_complex_loss_a_name] = make_complex_loss_a_arrow
        
        # Connect post to complex loss
        make_complex_loss_b_name = f"make_complex_loss_b_{t}_{v}"
        make_complex_loss_b_arrow = Arrow(
            start_box = box_dict[f"post_{t}_{v}"], 
            stop_box = box_dict[f"complex_loss_{t}_{v}"], 
            start_pos = post_to_complex_start_offset, 
            stop_pos = post_to_complex_stop_offset,
            text = "$D_{KL}$")
        arrow_dict[make_complex_loss_b_name] = make_complex_loss_b_arrow
        
        # Backprop hidden to post
        backprop_hidden_state_post_name = f"backprop_hidden_state_post_{t}_{v}"
        backprop_hidden_state_post_arrow = Arrow(
            start_box = box_dict[f"hidden_state_{t}_{v}"], 
            stop_box = box_dict[f"post_{t}_{v}"], 
            start_pos = backprop_hidden_to_post_start_offset, 
            stop_pos = backprop_hidden_to_post_stop_offset, 
            color = "red")
        arrow_dict[backprop_hidden_state_post_name] = backprop_hidden_state_post_arrow
        
        # Backprop complex loss to post
        backprop_complex_loss_post_name = f"backprop_complex_loss_post_{t}_{v}"
        backprop_complex_loss_post_arrow = Arrow(
            start_box = box_dict[f"complex_loss_{t}_{v}"], 
            stop_box = box_dict[f"post_{t}_{v}"], 
            start_pos = backprop_complex_loss_to_post_start_offset, 
            stop_pos = backprop_complex_loss_to_post_stop_offset, 
            color = "red")
        arrow_dict[backprop_complex_loss_post_name] = backprop_complex_loss_post_arrow
        
        # Backprop complex loss to prior
        backprop_complex_loss_prior_name = f"backprop_complex_loss_prior_{t}_{v}"
        backprop_complex_loss_prior_arrow = Arrow(
            start_box = box_dict[f"complex_loss_{t}_{v}"], 
            stop_box = box_dict[f"prior_{t}_{v}"], 
            start_pos = backprop_complex_loss_to_prior_start_offset, 
            stop_pos = backprop_complex_loss_to_prior_stop_offset, 
            color = "red")
        arrow_dict[backprop_complex_loss_prior_name] = backprop_complex_loss_prior_arrow
        
        # Backprop pred to hidden
        backprop_pred_hidden_state_name = f"backprop_pred_hidden_state_{t}_{v}"
        backprop_pred_hidden_state_arrow = Arrow(
            start_box = box_dict[f"pred_x_{t}_{v}"], 
            stop_box = box_dict[f"hidden_state_{t}_{v}"], 
            start_pos = backprop_pred_to_hidden_state_start_offset, 
            stop_pos = backprop_pred_to_hidden_state_stop_offset, 
            color = "red")
        arrow_dict[backprop_pred_hidden_state_name] = backprop_pred_hidden_state_arrow
        
        # Backprop accuracy_loss to pred
        backprop_accuracy_loss_pred_name = f"backprop_accuracy_loss_pred_{t}_{v}"
        backprop_accuracy_loss_pred_arrow = Arrow(
            start_box = box_dict[f"accuracy_loss_{t}_{v}"], 
            stop_box = box_dict[f"pred_x_{t}_{v}"], 
            start_pos = backprop_accuracy_loss_to_pred_start_offset, 
            stop_pos = backprop_accuracy_loss_to_pred_stop_offset, 
            color = "red")
        arrow_dict[backprop_accuracy_loss_pred_name] = backprop_accuracy_loss_pred_arrow
        
        # Backprop prior to hidden
        backprop_prior_hidden_state_name = f"backprop_prior_hidden_state_{t}_{v}"
        if(t == 0):
            backprop_prior_hidden_state_arrow = Arrow(
                start_box = box_dict[f"prior_{t}_{v}"], 
                stop_box = box_dict[f"hidden_state_m1_{v}"], 
                start_pos = backprop_prior_to_hidden_state_start_offset, 
                stop_pos = backprop_prior_to_hidden_state_stop_offset, 
                color = "red")
        else:
            backprop_prior_hidden_state_arrow = Arrow(
                start_box = box_dict[f"prior_{t}_{v}"], 
                stop_box = box_dict[f"hidden_state_{t-1}_{v+1}"], 
                start_pos = backprop_prior_to_hidden_state_start_offset, 
                stop_pos = backprop_prior_to_hidden_state_stop_offset, 
                color = "red")
        arrow_dict[backprop_prior_hidden_state_name] = backprop_prior_hidden_state_arrow
        
        # Backprop hidden to hidden
        backprop_hidden_state_hidden_state_name = f"backprop_hidden_state_hidden_state_{t}_{v}"
        if(t == 0):
            backprop_hidden_state_hidden_state_arrow = Arrow(
                start_box = box_dict[f"hidden_state_0_{v}"], 
                stop_box = box_dict[f"hidden_state_m1_{v}"], 
                start_pos = backprop_hidden_state_to_hidden_state_start_offset, 
                stop_pos = backprop_hidden_state_to_hidden_state_stop_offset, 
                color = "red")
        else:
            backprop_hidden_state_hidden_state_arrow = Arrow(
                start_box = box_dict[f"hidden_state_{t}_{v}"], 
                stop_box = box_dict[f"hidden_state_{t-1}_{v+1}"], 
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

post_0_0_side_text = Box(
    text="Initiate first $t=0$ posterior\ninner state $(\mu = 0, \sigma = 1)$",  
    pos=add_pos(box_dict["post_0_0"], (3, .1)),
    width=0,
    height=0)

slide_1 = Slide(
    slide_title = "1: Initiate",
    box_list = [
        box_dict["hidden_state_m1_0"], 
        box_dict["post_0_0"]],
    arrow_list = [],
    side_text_list = [
        hidden_state_m1_0_side_text, 
        post_0_0_side_text])



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



slide_list = [
    slide_1,
    slide_2,
    slide_3,
    slide_4,
    slide_5,
    slide_6,
    slide_7,
    slide_8,
    slide_9,
    slide_10,
    slide_11,
    slide_12,
    slide_13,
    slide_14]

min_x_pos, max_x_pos, min_y_pos, max_y_pos, center_pos = get_sizes(slide_list)

center = Box(
    text=r"",                 
    pos=center_pos,
    width=0,
    height=0)



title_text = Box(
    text="Slide Show for PVRNN Error Regression\n(Predictive-Coding-Inspired Variational RNN)",                 
    pos=center_pos,
    width=0,
    height=0)

slide_title = Slide(
    slide_title = "",
    box_list = [title_text],
    arrow_list = [])



intro = Box(
    text="It's important to note how\nPVRNN remakes previous variables.\nHow to read variables:",                 
    pos=add_pos(center, (0, 2)),
    width=0,
    height=0)

example = Box(
    text=r"$x^{\#3}_{t=5}$",                 
    pos=add_pos(center, (-1.5, 0)),
    width=default_width,
    height=.5)

hashtag_text = Box(
    text="Superscript with # is\nhow many times the variable\nhas been updated",
    pos=add_pos(example, (3, .5)),
    width=0,
    height=0)

t_text = Box(
    text="Subscript with $t=$ is\nwhich time-step the\nvariable exists in",
    pos=add_pos(example, (3, -.5)),
    width=0,
    height=0)

to_hashtag = Arrow(
    start_box=hashtag_text, 
    stop_box=example,
    start_pos = (-1.75, 0))

to_t = Arrow(
    start_box=t_text, 
    stop_box=example,
    start_pos = (-1.25, 0))

intro_end = Box(
    text="This example is the third estimate of observation 5.",                 
    pos=add_pos(center, (0, -1.5)),
    width=0,
    height=0)

slide_intro = Slide(
    slide_title = "Introduction",
    box_list = [
        center,
        intro,
        example,
        hashtag_text,
        t_text,
        intro_end],
    arrow_list = [
        to_hashtag,
        to_t])



slide_list = [slide_title, slide_intro] + slide_list



for slide in slide_list:
    slide.plot_slide(min_x_pos, max_x_pos, min_y_pos, max_y_pos, center_pos, axes = False)
    
    
    
    
    
    
    
    
    
    
    
# Step 15: Final overview
default_width = 1.5
prior_post_width = 4.25



hidden_to_hidden = (5, 0)

hidden_to_prior = (3.5, 1.6)
hidden_to_prior_stop_offset = (-1.5, .3)

hidden_to_post = (3.5, 1)
hidden_to_post_start_offset = (.6, 0)
hidden_to_post_stop_offset = (.2, 0)

post_to_complex = (2.5, 1.25)
post_to_complex_start_offset = (1.65, -.2)
post_to_complex_stop_offset = (.5, 0)

prior_to_complex_start_offset = (.4, 0)
prior_to_complex_stop_offset = (0, 0)

backprop_hidden_to_post_start_offset = (.1, 0)
backprop_hidden_to_post_stop_offset = (.5, 0)

backprop_complex_loss_to_post_start_offset = (.4, 0)
backprop_complex_loss_to_post_stop_offset = (.8, -.1)

backprop_complex_loss_to_prior_start_offset = (0, .1)
backprop_complex_loss_to_prior_stop_offset = (.3, 0)

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



for t in [-1, 0, 1, 2, 3]:
    t_name = t_name_maker(t)
    for v in [0, 1, 2, 3, 4]:
        hidden_state_name = f"hidden_state_{t_name}_{v}"
        hidden_state_box = make_box(
            box_type = "hidden_state", t = t, v = v, 
            pos = (0, 0) if t == -1 else add_pos(box_dict[f"hidden_state_{t_name_maker(t-1)}_{v}"], hidden_to_hidden))
        box_dict[hidden_state_name] = hidden_state_box
        
        if(t >= 0):
            prior_name = f"prior_{t}_{v}"
            prior_box = make_box(
                box_type = "prior", t = t, v = v, 
                pos = add_pos(box_dict[f"hidden_state_{t_name_maker(t-1)}_{v}"], hidden_to_prior), width = prior_post_width)
            box_dict[prior_name] = prior_box
            
            post_name = f"post_{t}_{v}"
            post_box = make_box(
                box_type = "post", t = t, v = v, 
                pos = add_pos(box_dict[f"hidden_state_{t_name_maker(t-1)}_{v}"], hidden_to_post), width = prior_post_width)
            box_dict[post_name] = post_box
            
            pred_x_name = f"pred_x_{t}_{v}"
            pred_x_box = make_box(
                box_type = "pred_x", t = t, v = v, 
                pos = add_pos(box_dict[f"hidden_state_{t}_{v}"], (0, -1)))
            box_dict[pred_x_name] = pred_x_box
            
            accuracy_loss_name = f"accuracy_loss_{t}_{v}"
            accuracy_loss_box = make_box(
                box_type = "accuracy_loss", t = t, v = v,
                pos = add_pos(box_dict[f"pred_x_{t}_{v}"], (0, -1)))
            box_dict[accuracy_loss_name] = accuracy_loss_box
            
            complex_loss_name = f"complex_loss_{t}_{v}"
            complex_box = make_box(
                box_type = "complex_loss", t = t, v = v,
                pos = add_pos(box_dict[f"post_{t}_{v}"], post_to_complex))
            box_dict[complex_loss_name] = complex_box
            
    if(t >= 0):
        real_x_name = f"real_x_{t}"
        real_x_box = make_box(
            box_type = "real_x", t = t, v = None,
            pos = add_pos(box_dict[f"accuracy_loss_{t}_0"], (0, -1)))
        box_dict[real_x_name] = real_x_box
            

            
arrow_dict = {}

for t in [0, 1, 2]:
    t_name = t_name_maker(t)
    for v in [0, 1, 2, 3]:
        # Connect hidden state to next prior
        make_prior_name = f"make_prior_{t}_{v}"
        if(t == 0):
            make_prior_arrow = Arrow(
                start_box = box_dict[f"hidden_state_{t_name_maker(t-1)}_{v}"], 
                stop_box = box_dict[f"prior_{t}_{v}"], 
                stop_pos = hidden_to_prior_stop_offset)
        elif(v > 0):
            make_prior_arrow = Arrow(
                start_box = box_dict[f"hidden_state_{t_name_maker(t-1)}_{v}"], 
                stop_box = box_dict[f"prior_{t}_{v-1}"], 
                stop_pos = hidden_to_prior_stop_offset)
        else:
            make_prior_arrow = Arrow(
                start_box = box_dict[f"hidden_state_{0}_{0}"], 
                stop_box = box_dict[f"prior_{0}_{0}"], 
                stop_pos = hidden_to_prior_stop_offset)
        arrow_dict[make_prior_name] = make_prior_arrow
        
        # Connect hidden_state to next hidden_state
        make_hidden_state_a_name = f"make_hidden_state_a_{t}_{v}"
        if(t == 0):
            make_hidden_state_a_arrow = Arrow(
                start_box = box_dict[f"hidden_state_{t_name_maker(t-1)}_{v}"], 
                stop_box = box_dict[f"hidden_state_{t}_{v}"])
        elif(v > 0):
            make_hidden_state_a_arrow = Arrow(
                start_box = box_dict[f"hidden_state_{t_name_maker(t-1)}_{v}"], 
                stop_box = box_dict[f"hidden_state_{t}_{v-1}"])
        else:
            make_hidden_state_a_arrow = Arrow(
                start_box = box_dict[f"hidden_state_{0}_{0}"], 
                stop_box = box_dict[f"hidden_state_{0}_{0}"])
        arrow_dict[make_hidden_state_a_name] = make_hidden_state_a_arrow
        
        # Connect post to hidden
        make_hidden_state_b_name = f"make_hidden_state_b_{t}_{v}"
        make_hidden_state_b_arrow = Arrow(
            start_box = box_dict[f"post_{t}_{v}"], 
            stop_box = box_dict[f"hidden_state_{t}_{v}"], 
            start_pos = hidden_to_post_start_offset, 
            stop_pos = hidden_to_post_stop_offset)
        arrow_dict[make_hidden_state_b_name] = make_hidden_state_b_arrow
        
        # Connect hidden to pred
        make_pred_x_0_0_name = f"make_pred_x_{t}_{v}"
        make_pred_x_arrow = Arrow(
            start_box = box_dict[f"hidden_state_{t}_{v}"], 
            stop_box = box_dict[f"pred_x_{t}_{v}"])
        arrow_dict[make_pred_x_0_0_name] = make_pred_x_arrow
        
        # Connect pred to accuracy loss
        make_accuracy_loss_a_name = f"make_accuracy_loss_a_{t}_{v}"
        make_accuracy_loss_a_arrow = Arrow(
            start_box = box_dict[f"pred_x_{t}_{v}"], 
            stop_box = box_dict[f"accuracy_loss_{t}_{v}"])
        arrow_dict[make_accuracy_loss_a_name] = make_accuracy_loss_a_arrow
        
        # Connect real to accuracy loss
        make_accuracy_loss_b_name = f"make_accuracy_loss_b_{t}_{v}"
        make_accuracy_loss_b_arrow = Arrow(
            start_box = box_dict[f"real_x_{t}"], 
            stop_box = box_dict[f"accuracy_loss_{t}_{v}"])
        arrow_dict[make_accuracy_loss_b_name] = make_accuracy_loss_b_arrow
        
        # Connect prior to complex loss
        make_complex_loss_a_name = f"make_complex_loss_a_{t}_{v}"
        make_complex_loss_a_arrow = Arrow(
            start_box = box_dict[f"prior_{t}_{v}"], 
            stop_box = box_dict[f"complex_loss_{t}_{v}"], 
            start_pos = prior_to_complex_start_offset, 
            stop_pos = prior_to_complex_stop_offset)
        arrow_dict[make_complex_loss_a_name] = make_complex_loss_a_arrow
        
        # Connect post to complex loss
        make_complex_loss_b_name = f"make_complex_loss_b_{t}_{v}"
        make_complex_loss_b_arrow = Arrow(
            start_box = box_dict[f"post_{t}_{v}"], 
            stop_box = box_dict[f"complex_loss_{t}_{v}"], 
            start_pos = post_to_complex_start_offset, 
            stop_pos = post_to_complex_stop_offset)
        arrow_dict[make_complex_loss_b_name] = make_complex_loss_b_arrow
        
        # Backprop hidden to post
        backprop_hidden_state_post_name = f"backprop_hidden_state_post_{t}_{v}"
        backprop_hidden_state_post_arrow = Arrow(
            start_box = box_dict[f"hidden_state_{t}_{v}"], 
            stop_box = box_dict[f"post_{t}_{v}"], 
            start_pos = backprop_hidden_to_post_start_offset, 
            stop_pos = backprop_hidden_to_post_stop_offset, 
            color = "red")
        arrow_dict[backprop_hidden_state_post_name] = backprop_hidden_state_post_arrow
        
        # Backprop complex loss to post
        backprop_complex_loss_post_name = f"backprop_complex_loss_post_{t}_{v}"
        backprop_complex_loss_post_arrow = Arrow(
            start_box = box_dict[f"complex_loss_{t}_{v}"], 
            stop_box = box_dict[f"post_{t}_{v}"], 
            start_pos = backprop_complex_loss_to_post_start_offset, 
            stop_pos = backprop_complex_loss_to_post_stop_offset, 
            color = "red")
        arrow_dict[backprop_complex_loss_post_name] = backprop_complex_loss_post_arrow
        
        # Backprop complex loss to prior
        backprop_complex_loss_prior_name = f"backprop_complex_loss_prior_{t}_{v}"
        backprop_complex_loss_prior_arrow = Arrow(
            start_box = box_dict[f"complex_loss_{t}_{v}"], 
            stop_box = box_dict[f"prior_{t}_{v}"], 
            start_pos = backprop_complex_loss_to_prior_start_offset, 
            stop_pos = backprop_complex_loss_to_prior_stop_offset, 
            color = "red")
        arrow_dict[backprop_complex_loss_prior_name] = backprop_complex_loss_prior_arrow
        
        # Backprop pred to hidden
        backprop_pred_hidden_state_name = f"backprop_pred_hidden_state_{t}_{v}"
        backprop_pred_hidden_state_arrow = Arrow(
            start_box = box_dict[f"pred_x_{t}_{v}"], 
            stop_box = box_dict[f"hidden_state_{t}_{v}"], 
            start_pos = backprop_pred_to_hidden_state_start_offset, 
            stop_pos = backprop_pred_to_hidden_state_stop_offset, 
            color = "red")
        arrow_dict[backprop_pred_hidden_state_name] = backprop_pred_hidden_state_arrow
        
        # Backprop accuracy_loss to pred
        backprop_accuracy_loss_pred_name = f"backprop_accuracy_loss_pred_{t}_{v}"
        backprop_accuracy_loss_pred_arrow = Arrow(
            start_box = box_dict[f"accuracy_loss_{t}_{v}"], 
            stop_box = box_dict[f"pred_x_{t}_{v}"], 
            start_pos = backprop_accuracy_loss_to_pred_start_offset, 
            stop_pos = backprop_accuracy_loss_to_pred_stop_offset, 
            color = "red")
        arrow_dict[backprop_accuracy_loss_pred_name] = backprop_accuracy_loss_pred_arrow
        
        # Backprop prior to hidden
        backprop_prior_hidden_state_name = f"backprop_prior_hidden_state_{t}_{v}"
        if(t == 0):
            backprop_prior_hidden_state_arrow = Arrow(
                start_box = box_dict[f"prior_{t}_{v}"], 
                stop_box = box_dict[f"hidden_state_m1_{v}"], 
                start_pos = backprop_prior_to_hidden_state_start_offset, 
                stop_pos = backprop_prior_to_hidden_state_stop_offset, 
                color = "red")
        else:
            backprop_prior_hidden_state_arrow = Arrow(
                start_box = box_dict[f"prior_{t}_{v}"], 
                stop_box = box_dict[f"hidden_state_{t-1}_{v+1}"], 
                start_pos = backprop_prior_to_hidden_state_start_offset, 
                stop_pos = backprop_prior_to_hidden_state_stop_offset, 
                color = "red")
        arrow_dict[backprop_prior_hidden_state_name] = backprop_prior_hidden_state_arrow
        
        # Backprop hidden to hidden
        backprop_hidden_state_hidden_state_name = f"backprop_hidden_state_hidden_state_{t}_{v}"
        if(t == 0):
            backprop_hidden_state_hidden_state_arrow = Arrow(
                start_box = box_dict[f"hidden_state_0_{v}"], 
                stop_box = box_dict[f"hidden_state_m1_{v}"], 
                start_pos = backprop_hidden_state_to_hidden_state_start_offset, 
                stop_pos = backprop_hidden_state_to_hidden_state_stop_offset, 
                color = "red")
        else:
            backprop_hidden_state_hidden_state_arrow = Arrow(
                start_box = box_dict[f"hidden_state_{t}_{v}"], 
                stop_box = box_dict[f"hidden_state_{t-1}_{v+1}"], 
                start_pos = backprop_hidden_state_to_hidden_state_start_offset, 
                stop_pos = backprop_hidden_state_to_hidden_state_stop_offset, 
                color = "red")
        arrow_dict[backprop_hidden_state_hidden_state_name] = backprop_hidden_state_hidden_state_arrow
        
and_so_on_side_text = Box(
    text="And so on",  
    pos=(0, 3),
    width=0,
    height=0
)

# Gather all final (v=2) boxes for t = -1, 0, 1, 2
boxes_15 = [
    # The second-update hidden state at t = -1
    box_dict["hidden_state_m1_2"],

    # t = 0
    box_dict["hidden_state_0_2"],
    box_dict["prior_0_2"],
    box_dict["post_0_2"],
    box_dict["pred_x_0_2"],
    box_dict["accuracy_loss_0_2"],
    box_dict["real_x_0"],
    box_dict["complex_loss_0_2"],

    # t = 1
    box_dict["hidden_state_1_1"],
    box_dict["prior_1_1"],
    box_dict["post_1_1"],
    box_dict["pred_x_1_1"],
    box_dict["accuracy_loss_1_1"],
    box_dict["real_x_1"],
    box_dict["complex_loss_1_1"],

    # t = 2
    box_dict["hidden_state_2_0"],
    box_dict["prior_2_0"],
    box_dict["post_2_0"],
    box_dict["pred_x_2_0"],
    box_dict["accuracy_loss_2_0"],
    box_dict["real_x_2"],
    box_dict["complex_loss_2_0"]
]

# Gather all arrows for v=2, t = 0,1,2 (both forward and backprop)
arrows_15 = [
    # --- t=0 Forward pass ---
    arrow_dict["make_prior_0_2"],
    arrow_dict["make_hidden_state_a_0_2"], 
    arrow_dict["make_hidden_state_b_0_2"],
    arrow_dict["make_pred_x_0_2"],
    arrow_dict["make_accuracy_loss_a_0_2"], 
    arrow_dict["make_accuracy_loss_b_0_2"],
    arrow_dict["make_complex_loss_a_0_2"], 
    arrow_dict["make_complex_loss_b_0_2"],

    # --- t=0 Backprop ---
    #arrow_dict["backprop_hidden_state_post_0_2"],
    #arrow_dict["backprop_complex_loss_post_0_2"],
    #arrow_dict["backprop_complex_loss_prior_0_2"],
    #arrow_dict["backprop_pred_hidden_state_0_2"],
    #arrow_dict["backprop_accuracy_loss_pred_0_2"],
    #arrow_dict["backprop_prior_hidden_state_0_2"],
    #arrow_dict["backprop_hidden_state_hidden_state_0_2"],

    # --- t=1 Forward pass ---
    arrow_dict["make_prior_1_2"],
    arrow_dict["make_hidden_state_a_1_2"],
    arrow_dict["make_hidden_state_b_1_1"],
    arrow_dict["make_pred_x_1_1"],
    arrow_dict["make_accuracy_loss_a_1_1"],
    arrow_dict["make_accuracy_loss_b_1_1"],
    arrow_dict["make_complex_loss_a_1_1"],
    arrow_dict["make_complex_loss_b_1_1"],

    # --- t=1 Backprop ---
    #arrow_dict["backprop_hidden_state_post_1_2"],
    #arrow_dict["backprop_complex_loss_post_1_2"],
    #arrow_dict["backprop_complex_loss_prior_1_2"],
    #arrow_dict["backprop_pred_hidden_state_1_2"],
    #arrow_dict["backprop_accuracy_loss_pred_1_2"],
    #arrow_dict["backprop_prior_hidden_state_1_2"],
    #arrow_dict["backprop_hidden_state_hidden_state_1_2"],

    # --- t=2 Forward pass ---
    arrow_dict["make_prior_2_1"],
    arrow_dict["make_hidden_state_a_2_1"],
    arrow_dict["make_hidden_state_b_2_0"],
    arrow_dict["make_pred_x_2_0"],
    arrow_dict["make_accuracy_loss_a_2_0"],
    arrow_dict["make_accuracy_loss_b_2_0"],
    arrow_dict["make_complex_loss_a_2_0"],
    arrow_dict["make_complex_loss_b_2_0"],

    # --- t=2 Backprop ---
    #arrow_dict["backprop_hidden_state_post_2_2"],
    #arrow_dict["backprop_complex_loss_post_2_2"],
    #arrow_dict["backprop_complex_loss_prior_2_2"],
    #arrow_dict["backprop_pred_hidden_state_2_2"],
    #arrow_dict["backprop_accuracy_loss_pred_2_2"],
    #arrow_dict["backprop_prior_hidden_state_2_2"],
    #arrow_dict["backprop_hidden_state_hidden_state_2_2"],
]

slide_15 = Slide(
    slide_title="15: Repeat further",
    box_list=boxes_15,
    arrow_list=arrows_15,
    side_text_list=[and_so_on_side_text]
)

min_x_pos, max_x_pos, min_y_pos, max_y_pos, center_pos = get_sizes([slide_15])

slide_15.plot_slide(min_x_pos, max_x_pos, min_y_pos, max_y_pos, center_pos, axes = False)