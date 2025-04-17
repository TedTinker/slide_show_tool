#%% 
from slide_show_tool import *



default_width = .75
prior_post_width = 2.5



hidden_to_hidden = (3.5, 0)

hidden_to_prior = (2.75, 1.6)
hidden_to_prior_stop_offset = (-.8, .2)
hidden_to_post_stop_offset = (-.8, .2)

action_to_prior_stop_offset = (-.8, 0)
action_to_post_stop_offset = (-.8, 0)
action_to_post_start_offset = (0, 0)
action_to_post_stop_offset = (0, 0)

obs_to_post_stop_offset = (0, 0)

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
    text=None,
    pos=(0, 0),
    width=default_width,
    height=0.5):
    if text is None:
        if box_type == "hidden_state":
            text = fr"$d_{{t={t}}}$"
        elif box_type == "action":
            text = fr"$a_{{t={t}}}$"
        elif box_type == "post":
            text = fr"$(\mu^q_{{t={t}}}, \sigma^q_{{t={t}}}) \rightarrow z^q_{{t={t}}}$"
        elif box_type == "prior":
            text = fr"$(\mu^p_{{t={t}}}, \sigma^p_{{t={t}}}) \rightarrow z^p_{{t={t}}}$"
        elif box_type == "pred_x":
            text = fr"$x_{{t={t}}}$"
        elif box_type == "accuracy_loss":
            text = fr"$L^{{A}}_{{t={t}}}$"
        elif box_type == "real_x":
            text = fr"$x_{{t={t}}}$"
        elif box_type == "complex_loss":
            text = fr"$L^{{C}}_{{t={t}}}$"
        else:
            text = f"Unknown box_type={box_type} (t={t}, v={v})"
    return Box(
        text=text,
        pos=pos,
        width=width,
        height=height)



for t in [-1, 0, 1, 2, 3]:
    t_name = t_name_maker(t)

    hidden_state_name = f"hidden_state_{t_name}"
    hidden_state_box = make_box(
        box_type = "hidden_state", t = t, 
        pos = (0, 0) if t == -1 else add_pos(box_dict[f"hidden_state_{t_name_maker(t-1)}"], hidden_to_hidden))
    box_dict[hidden_state_name] = hidden_state_box
    
    action_name = f"action_{t_name}"
    action_box = make_box(
        box_type = "action", t = t, 
        pos = add_pos(box_dict[hidden_state_name], (0, -1)) if t == -1 else add_pos(box_dict[f"action_{t_name_maker(t-1)}"], hidden_to_hidden))
    box_dict[action_name] = action_box
    
    if(t >= 0):
        prior_name = f"prior_{t}"
        prior_box = make_box(
            box_type = "prior", t = t, 
            pos = add_pos(box_dict[f"hidden_state_{t_name_maker(t-1)}"], hidden_to_prior), width = prior_post_width)
        box_dict[prior_name] = prior_box
        
        post_name = f"post_{t}"
        post_box = make_box(
            box_type = "post", t = t, 
            pos = add_pos(box_dict[f"hidden_state_{t_name_maker(t-1)}"], hidden_to_post), width = prior_post_width)
        box_dict[post_name] = post_box
        
        pred_x_name = f"pred_x_{t}"
        pred_x_box = make_box(
            box_type = "pred_x", t = t, 
            pos = add_pos(box_dict[f"hidden_state_{t}"], (-1, -1)))
        box_dict[pred_x_name] = pred_x_box
        
        accuracy_loss_name = f"accuracy_loss_{t}"
        accuracy_loss_box = make_box(
            box_type = "accuracy_loss", t = t, 
            pos = add_pos(box_dict[f"pred_x_{t}"], (0, -1)))
        box_dict[accuracy_loss_name] = accuracy_loss_box
        
        complex_loss_name = f"complex_loss_{t}"
        complex_box = make_box(
            box_type = "complex_loss", t = t, 
            pos = add_pos(box_dict[f"post_{t}"], post_to_complex))
        box_dict[complex_loss_name] = complex_box
            
    if(t >= 0):
        real_x_name = f"real_x_{t}"
        real_x_box = make_box(
            box_type = "real_x", t = t, 
            pos = add_pos(box_dict[f"accuracy_loss_{t}"], (0, -1)))
        box_dict[real_x_name] = real_x_box
            

            
arrow_dict = {}

for t in [0, 1, 2]:
    t_name = t_name_maker(t)

    # Connect hidden state to next prior
    make_prior_a_name = f"make_prior_a_{t}"
    if(t == 0):
        make_prior_a_arrow = Arrow(
            start_box = box_dict[f"hidden_state_{t_name_maker(t-1)}"], 
            stop_box = box_dict[f"prior_{t}"], 
            stop_pos = hidden_to_prior_stop_offset,
            text = "$w_{d,p}$")
    arrow_dict[make_prior_a_name] = make_prior_a_arrow
    
    # Connect action to next prior
    make_prior_b_name = f"make_prior_b_{t}"
    if(t == 0):
        make_prior_b_arrow = Arrow(
            start_box = box_dict[f"action_{t_name_maker(t-1)}"], 
            stop_box = box_dict[f"prior_{t}"], 
            stop_pos = action_to_prior_stop_offset,
            text = "$w_{a,p}$")
    arrow_dict[make_prior_b_name] = make_prior_b_arrow
    
    # Connect hidden state to next post
    make_post_a_name = f"make_post_a_{t}"
    if(t == 0):
        make_post_a_arrow = Arrow(
            start_box = box_dict[f"hidden_state_{t_name_maker(t-1)}"], 
            stop_box = box_dict[f"post_{t}"], 
            stop_pos = hidden_to_post_stop_offset,
            text = "$w_{d,q}$")
    arrow_dict[make_post_a_name] = make_post_a_arrow
    
    # Connect action to next post
    make_post_b_name = f"make_post_b_{t}"
    if(t == 0):
        make_post_b_arrow = Arrow(
            start_box = box_dict[f"action_{t_name_maker(t-1)}"], 
            stop_box = box_dict[f"post_{t}"], 
            stop_pos = action_to_post_stop_offset,
            text = "$w_{a,q}$")
    arrow_dict[make_post_b_name] = make_post_b_arrow
    
    # Connect real_x to post
    make_post_c_name = f"make_post_c_{t}"
    if(t == 0):
        make_post_c_arrow = Arrow(
            start_box = box_dict[f"real_x_{t_name_maker(t)}"], 
            stop_box = box_dict[f"post_{t}"], 
            stop_pos = obs_to_post_stop_offset,
            text = "$w_{x,q}$")
    arrow_dict[make_post_c_name] = make_post_c_arrow
    
    # Connect hidden_state to next hidden_state
    make_hidden_state_a_name = f"make_hidden_state_a_{t}"
    if(t == 0):
        make_hidden_state_a_arrow = Arrow(
            start_box = box_dict[f"hidden_state_{t_name_maker(t-1)}"], 
            stop_box = box_dict[f"hidden_state_{t}"],
            text = "$w_{d,d}$")
    arrow_dict[make_hidden_state_a_name] = make_hidden_state_a_arrow
    
    # Connect post to hidden
    make_hidden_state_b_name = f"make_hidden_state_b_{t}"
    make_hidden_state_b_arrow = Arrow(
        start_box = box_dict[f"post_{t}"], 
        stop_box = box_dict[f"hidden_state_{t}"], 
        start_pos = hidden_to_post_start_offset, 
        stop_pos = hidden_to_post_stop_offset,
        text = "$w_{q,d}$")
    arrow_dict[make_hidden_state_b_name] = make_hidden_state_b_arrow
    
    # Connect action to hidden
    make_hidden_state_c_name = f"make_hidden_state_c_{t}"
    make_hidden_state_c_arrow = Arrow(
        start_box = box_dict[f"action_{t_name_maker(t-1)}"], 
        stop_box = box_dict[f"hidden_state_{t}"], 
        start_pos = action_to_post_start_offset, 
        stop_pos = action_to_post_stop_offset,
        text = "$w_{a,d}$")
    arrow_dict[make_hidden_state_c_name] = make_hidden_state_c_arrow
    
    # Connect hidden to pred
    make_pred_x_a_name = f"make_pred_x_a_{t}"
    make_pred_x_a_arrow = Arrow(
        start_box = box_dict[f"hidden_state_{t_name_maker(t-1)}"], 
        stop_box = box_dict[f"pred_x_{t}"],
        text = "$w_{d,x}$")
    arrow_dict[make_pred_x_a_name] = make_pred_x_a_arrow
    
    # Connect action to pred
    make_pred_x_b_name = f"make_pred_x_b_{t}"
    make_pred_x_b_arrow = Arrow(
        start_box = box_dict[f"action_{t_name_maker(t-1)}"], 
        stop_box = box_dict[f"pred_x_{t}"],
        text = "$w_{a,x}$")
    arrow_dict[make_pred_x_b_name] = make_pred_x_b_arrow
    
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
    
    # Connect prior to complex loss
    make_complex_loss_a_name = f"make_complex_loss_a_{t}"
    make_complex_loss_a_arrow = Arrow(
        start_box = box_dict[f"prior_{t}"], 
        stop_box = box_dict[f"complex_loss_{t}"], 
        start_pos = prior_to_complex_start_offset, 
        stop_pos = prior_to_complex_stop_offset)
    arrow_dict[make_complex_loss_a_name] = make_complex_loss_a_arrow
    
    # Connect post to complex loss
    make_complex_loss_b_name = f"make_complex_loss_b_{t}"
    make_complex_loss_b_arrow = Arrow(
        start_box = box_dict[f"post_{t}"], 
        stop_box = box_dict[f"complex_loss_{t}"], 
        start_pos = post_to_complex_start_offset, 
        stop_pos = post_to_complex_stop_offset,
        text = "$D_{KL}$")
    arrow_dict[make_complex_loss_b_name] = make_complex_loss_b_arrow
    
    # Backprop hidden to post
    backprop_hidden_state_post_name = f"backprop_hidden_state_post_{t}"
    backprop_hidden_state_post_arrow = Arrow(
        start_box = box_dict[f"hidden_state_{t}"], 
        stop_box = box_dict[f"post_{t}"], 
        start_pos = backprop_hidden_to_post_start_offset, 
        stop_pos = backprop_hidden_to_post_stop_offset, 
        color = "red")
    arrow_dict[backprop_hidden_state_post_name] = backprop_hidden_state_post_arrow
    
    # Backprop complex loss to post
    backprop_complex_loss_post_name = f"backprop_complex_loss_post_{t}"
    backprop_complex_loss_post_arrow = Arrow(
        start_box = box_dict[f"complex_loss_{t}"], 
        stop_box = box_dict[f"post_{t}"], 
        start_pos = backprop_complex_loss_to_post_start_offset, 
        stop_pos = backprop_complex_loss_to_post_stop_offset, 
        color = "red")
    arrow_dict[backprop_complex_loss_post_name] = backprop_complex_loss_post_arrow
    
    # Backprop complex loss to prior
    backprop_complex_loss_prior_name = f"backprop_complex_loss_prior_{t}"
    backprop_complex_loss_prior_arrow = Arrow(
        start_box = box_dict[f"complex_loss_{t}"], 
        stop_box = box_dict[f"prior_{t}"], 
        start_pos = backprop_complex_loss_to_prior_start_offset, 
        stop_pos = backprop_complex_loss_to_prior_stop_offset, 
        color = "red")
    arrow_dict[backprop_complex_loss_prior_name] = backprop_complex_loss_prior_arrow
    
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
    
    # Backprop prior to hidden
    backprop_prior_hidden_state_name = f"backprop_prior_hidden_state_{t}"
    if(t == 0):
        backprop_prior_hidden_state_arrow = Arrow(
            start_box = box_dict[f"prior_{t}"], 
            stop_box = box_dict[f"hidden_state_m1"], 
            start_pos = backprop_prior_to_hidden_state_start_offset, 
            stop_pos = backprop_prior_to_hidden_state_stop_offset, 
            color = "red")
    else:
        backprop_prior_hidden_state_arrow = Arrow(
            start_box = box_dict[f"prior_{t}"], 
            stop_box = box_dict[f"hidden_state_{t-1}"], 
            start_pos = backprop_prior_to_hidden_state_start_offset, 
            stop_pos = backprop_prior_to_hidden_state_stop_offset, 
            color = "red")
    arrow_dict[backprop_prior_hidden_state_name] = backprop_prior_hidden_state_arrow
    
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
hidden_state_m1_side_text = Box(
    text="Initiate hidden state/context/\nlatent state as zeroes",  
    pos=add_pos(box_dict["hidden_state_m1"], (2.5, -.1)),
    width=0,
    height=0)

action_m1_side_text = Box(
    text="Initiate previous action as zeroes",  
    pos=add_pos(box_dict["action_m1"], (2.5, -.1)),
    width=0,
    height=0)

slide_1 = Slide(
    slide_title = "1: Initiate",
    box_list = [
        box_dict["hidden_state_m1"], 
        box_dict["action_m1"]],
    arrow_list = [],
    side_text_list = [
        hidden_state_m1_side_text, 
        action_m1_side_text])



# Step 2
prior_0_side_text = Box(
    text="Generate $t=0$\nprior inner state",  
    pos=add_pos(box_dict["prior_0"], (2.5, 0)),
    width=0,
    height=0)

slide_2 = Slide(
    slide_title = "2: Prior inner state",
    box_list = slide_1.box_list + [
        box_dict["prior_0"]],
    arrow_list = [
        arrow_dict["make_prior_a_0"],
        arrow_dict["make_prior_b_0"]],
    side_text_list = [
        prior_0_side_text])



# Step 3
post_0_side_text = Box(
    text="Generate $t=0$\nposterior inner state",  
    pos=add_pos(box_dict["post_0"], (2.5, 0)),
    width=0,
    height=0)

slide_3 = Slide(
    slide_title = "3: Posterior inner state",
    box_list = slide_2.box_list + [
        box_dict["real_x_0"],
        box_dict["post_0"]],
    arrow_list = slide_2.arrow_list + [
        arrow_dict["make_post_a_0"],
        arrow_dict["make_post_b_0"],
        arrow_dict["make_post_c_0"]],
    side_text_list = [
        post_0_side_text])



# Step 4
hidden_state_0_side_text = Box(
    text="Make new hidden state",  
    pos=add_pos(box_dict["hidden_state_0"], (2, 0)),
    width=0,
    height=0)

action_0_side_text = Box(
    text="Make new action",  
    pos=add_pos(box_dict["action_0"], (2, 0)),
    width=0,
    height=0)

slide_4 = Slide(
    slide_title = "4: New hidden state",
    box_list = slide_3.box_list + [
        box_dict["hidden_state_0"],
        box_dict["action_0"]],
    arrow_list = slide_3.arrow_list + [
        arrow_dict["make_hidden_state_a_0"],
        arrow_dict["make_hidden_state_b_0"],
        arrow_dict["make_hidden_state_c_0"]],
    side_text_list = [
        hidden_state_0_side_text,
        action_0_side_text])



# Step 5
pred_x_1_side_text = Box(
    text="Predict next observation",  
    pos=add_pos(box_dict["pred_x_1"], (0, -.75)),
    width=0,
    height=0)

slide_5 = Slide(
    slide_title = "5: Predict observation",
    box_list = slide_4.box_list + [
        box_dict["pred_x_1"]],
    arrow_list = slide_4.arrow_list + [
        arrow_dict["make_pred_x_a_1"],
        arrow_dict["make_pred_x_b_1"]],
    side_text_list = [
        pred_x_1_side_text])



# Step 6
slide_6 = Slide(
    slide_title = "6: Get loss values",
    box_list = slide_5.box_list + [
        box_dict["real_x_1"],
        box_dict["accuracy_loss_1"],
        box_dict["complex_loss_0"]],
    arrow_list = slide_5.arrow_list + [
        arrow_dict["make_pred_x_a_1"],
        arrow_dict["make_pred_x_b_1"],
        arrow_dict["make_accuracy_loss_a_1"], 
        arrow_dict["make_accuracy_loss_b_1"],
        arrow_dict["make_complex_loss_a_0"],
        arrow_dict["make_complex_loss_b_0"]],
    side_text_list = [])




slide_list = [
    slide_1,
    slide_2,
    slide_3,
    slide_4,
    slide_5,
    slide_6,
    #slide_7,
    #slide_8,
    #slide_9,
    #slide_10,
    #slide_11,
    #slide_12,
    #slide_13,
    #slide_14
    ]

min_x_pos, max_x_pos, min_y_pos, max_y_pos, center_pos = get_sizes(slide_list)

center = Box(
    text=r"",                 
    pos=center_pos,
    width=0,
    height=0)






for slide in slide_list:
    slide.plot_slide(min_x_pos, max_x_pos, min_y_pos, max_y_pos, center_pos, axes = False)
    
    
    
    
    
    
    
    
    
    