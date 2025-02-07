#%% 
from slide_show_tool import *



hidden_to_hidden = (2.5, 0)
hidden_to_prior = (.75, 1.6)
hidden_to_post = (1.75, 1)
post_to_complex = (.25, 1.25)

default_width = .75
prior_post_width = 2.3



def make_hidden_state_box(t, v, pos=(0, 0), width=default_width, height=0.5):
    if t < 0:
        t_name = f"m{-t}"
    else:
        t_name = str(t)
    text_string = fr"$d^{{\#{v}}}_{{t={t}}}$"
    hidden_state_box = Box(
        text=text_string,
        pos=pos,
        width=width,
        height=height)
    return(hidden_state_box)
    
hidden_state_m1_0 = make_hidden_state_box(-1, 0)
hidden_state_m1_1 = make_hidden_state_box(-1, 1)
hidden_state_m1_2 = make_hidden_state_box(-1, 2)

hidden_state_0_0 = make_hidden_state_box(0, 0, pos = add_pos(hidden_state_m1_0, hidden_to_hidden))
hidden_state_0_1 = make_hidden_state_box(0, 1, pos = add_pos(hidden_state_m1_0, hidden_to_hidden))
hidden_state_0_2 = make_hidden_state_box(0, 2, pos = add_pos(hidden_state_m1_0, hidden_to_hidden))

hidden_state_1_0 = make_hidden_state_box(1, 0, pos = add_pos(hidden_state_0_0, hidden_to_hidden))
hidden_state_1_1 = make_hidden_state_box(1, 1, pos = add_pos(hidden_state_0_0, hidden_to_hidden))
hidden_state_1_2 = make_hidden_state_box(1, 2, pos = add_pos(hidden_state_0_0, hidden_to_hidden))

hidden_state_2_0 = make_hidden_state_box(2, 0, pos = add_pos(hidden_state_1_0, hidden_to_hidden))
hidden_state_2_1 = make_hidden_state_box(2, 1, pos = add_pos(hidden_state_1_0, hidden_to_hidden))
hidden_state_2_2 = make_hidden_state_box(2, 2, pos = add_pos(hidden_state_1_0, hidden_to_hidden))




# Step 1


hidden_state_m1_0_side_text = Box(
    text="Initiate hidden state/context/\nlatent state as zeroes",  
    pos=add_pos(hidden_state_m1_0, (2.5, -.1)),
    width=0,
    height=0)

post_0_0 = Box(
    text=r"$(\mu^{q,\#0}_{t=0}, \sigma^{q,\#0}_{t=0}) \rightarrow z^{q,\#0}_{t=0}$",  
    pos=add_pos(hidden_state_m1_0, hidden_to_post),
    width=prior_post_width,
    height=0.5)

post_0_0_side_text = Box(
    text="Initiate first $t=0$ posterior\ninner state $(\mu = 0, \sigma = 1)$",  
    pos=add_pos(post_0_0, (3, .1)),
    width=0,
    height=0)



slide_1 = Slide(
    slide_title = 1,
    box_list = [
        hidden_state_m1_0, hidden_state_m1_0_side_text, 
        post_0_0, post_0_0_side_text],
    arrow_list = [])



# Step 2
prior_0_0 = Box(
    text=r"$(\mu^{p,\#0}_{t=0}, \sigma^{p,\#0}_{t=0}) \rightarrow z^{p,\#0}_{t=0}$",  
    pos=add_pos(hidden_state_m1_0, hidden_to_prior),
    width=prior_post_width,
    height=0.5)

prior_0_0_side_text = Box(
    text="Generate first $t=0$\nprior inner state",  
    pos=add_pos(prior_0_0, (0, 1)),
    width=0,
    height=0)

make_prior_0_0 = Arrow(
    start_box=hidden_state_m1_0, 
    stop_box=prior_0_0)



slide_2 = Slide(
    slide_title = 2,
    box_list = [
        hidden_state_m1_0, 
        post_0_0,
        prior_0_0, prior_0_0_side_text],
    arrow_list = [
        make_prior_0_0])



# Step 3
hidden_state_0_0_side_text = Box(
    text="Generate first\n$t=0$ hidden state",  
    pos=add_pos(hidden_state_0_0, (0, -1)),
    width=0,
    height=0)

make_hidden_state_0_0_a = Arrow(
    start_box=hidden_state_m1_0, 
    stop_box=hidden_state_0_0)

make_hidden_state_0_0_b = Arrow(
    start_box=post_0_0, 
    stop_box=hidden_state_0_0)



slide_3 = Slide(
    slide_title = 3,
    box_list = [
        hidden_state_m1_0, 
        post_0_0,
        prior_0_0,
        hidden_state_0_0, hidden_state_0_0_side_text],
    arrow_list = [
        make_prior_0_0,
        make_hidden_state_0_0_a, make_hidden_state_0_0_b])



# Step 4
pred_x_0_0 = Box(
    text=r"$x^{\#0}_{t=0}$",                 
    pos=add_pos(hidden_state_0_0, (0, -1)),
    width=default_width,
    height=0.5)

pred_x_0_0_side_text = Box(
    text="Generate first estimation of\n$t=0$ observation",  
    pos=add_pos(pred_x_0_0, (0, -1)),
    width=0,
    height=0)

make_pred_x_0_0 = Arrow(
    start_box=hidden_state_0_0, 
    stop_box=pred_x_0_0)



slide_4 = Slide(
    slide_title = 4,
    box_list = [
        hidden_state_m1_0, 
        post_0_0,
        prior_0_0,
        hidden_state_0_0, 
        pred_x_0_0, pred_x_0_0_side_text],
    arrow_list = [
        make_prior_0_0,
        make_hidden_state_0_0_a, make_hidden_state_0_0_b,
        make_pred_x_0_0])



# Step 5
accuracy_loss_0_0 = Box(
    text=r"$L^{A, \#0}_{t=0}$",                 
    pos=add_pos(pred_x_0_0, (0, -1)),
    width=default_width,
    height=0.5)

accuracy_loss_0_0_side_text = Box(
    text="First $t=0$ Error, Loss (Accuracy)",  
    pos=(-.25, -2),
    width=0,
    height=0)

real_x_0 = Box(
    text=r"$x_{t=0}$",                 
    pos=add_pos(accuracy_loss_0_0, (0, -1)),
    width=default_width,
    height=0.5)

real_x_0_side_text = Box(
    text="Real $t=0$ observation (target)",  
    pos=(-.25, -3),
    width=0,
    height=0)

make_accuracy_loss_0_0_a = Arrow(
    start_box=pred_x_0_0, 
    stop_box=accuracy_loss_0_0)

make_accuracy_loss_0_0_b = Arrow(
    start_box=real_x_0, 
    stop_box=accuracy_loss_0_0)



slide_5 = Slide(
    slide_title = 5,
    box_list = [
        hidden_state_m1_0, 
        post_0_0,
        prior_0_0,
        hidden_state_0_0, 
        pred_x_0_0, 
        accuracy_loss_0_0, accuracy_loss_0_0_side_text,
        real_x_0, real_x_0_side_text],
    arrow_list = [
        make_prior_0_0,
        make_hidden_state_0_0_a, make_hidden_state_0_0_b,
        make_pred_x_0_0,
        make_accuracy_loss_0_0_a, make_accuracy_loss_0_0_b])



# Step 6
complex_loss_0_0 = Box(
    text=r"$L^{C, \#0}_{t=0}$",                 
    pos=add_pos(post_0_0, post_to_complex),
    width=default_width,
    height=0.5)

complex_loss_0_0_side_text = Box(
    text="First $t=0$ prior/posterior\nDKL, Loss (Complexity)",  
    pos=add_pos(complex_loss_0_0, (0, 1)),
    width=0,
    height=0)

make_complex_loss_0_0_a = Arrow(
    start_box=prior_0_0, 
    stop_box=complex_loss_0_0)

make_complex_loss_0_0_b = Arrow(
    start_box=post_0_0, 
    stop_box=complex_loss_0_0)



slide_6 = Slide(
    slide_title = 6,
    box_list = [
        hidden_state_m1_0, 
        post_0_0,
        prior_0_0,
        hidden_state_0_0, 
        pred_x_0_0, 
        accuracy_loss_0_0,
        real_x_0,
        complex_loss_0_0, complex_loss_0_0_side_text],
    arrow_list = [
        make_prior_0_0,
        make_hidden_state_0_0_a, make_hidden_state_0_0_b,
        make_pred_x_0_0,
        make_accuracy_loss_0_0_a, make_accuracy_loss_0_0_b,
        make_complex_loss_0_0_a, make_complex_loss_0_0_b])



# Step 7
first_backprop_side_text = Box(
    text="With backpropogation,\nupdate hidden state $t=-1$ and\nposterior inner state for t = 0",  
    pos=(2, 4),
    width=0,
    height=0)

slide_7 = Slide(
    slide_title = 7,
    box_list = [
        hidden_state_m1_0, 
        post_0_0,
        prior_0_0,
        hidden_state_0_0, 
        pred_x_0_0, 
        accuracy_loss_0_0,
        real_x_0,
        complex_loss_0_0, 
        first_backprop_side_text],
    arrow_list = [
        make_prior_0_0,
        make_hidden_state_0_0_a, make_hidden_state_0_0_b,
        make_pred_x_0_0,
        make_accuracy_loss_0_0_a, make_accuracy_loss_0_0_b,
        make_complex_loss_0_0_a, make_complex_loss_0_0_b])



# Step 8
first_backprop_complete_side_text = Box(
    text="Values updated",  
    pos=(2, 3),
    width=0,
    height=0)

post_0_1 = Box(
    text=r"$(\mu^{q,\#1}_{t=0}, \sigma^{q,\#1}_{t=0}) \rightarrow z^{q,\#1}_{t=0}$",  
    pos=add_pos(hidden_state_m1_1, hidden_to_post),
    width=prior_post_width,
    height=0.5)



slide_8 = Slide(
    slide_title = 8,
    box_list = [
        first_backprop_complete_side_text,
        hidden_state_m1_1,
        post_0_1],
    arrow_list = [])



# Step 9
do_that_again_side_text = Box(
    text="Perform the previous steps\nfor the second time",  
    pos=(2, 3),
    width=0,
    height=0)

prior_0_1 = Box(
    text=r"$(\mu^{p,\#1}_{t=0}, \sigma^{p,\#1}_{t=0}) \rightarrow z^{p,\#1}_{t=0}$",  
    pos=add_pos(hidden_state_m1_1, hidden_to_prior),
    width=prior_post_width,
    height=0.5)

make_prior_0_1 = Arrow(
    start_box=hidden_state_m1_1, 
    stop_box=prior_0_1)

make_hidden_state_0_1_a = Arrow(
    start_box=hidden_state_m1_1, 
    stop_box=hidden_state_0_1)

make_hidden_state_0_1_b = Arrow(
    start_box=post_0_1, 
    stop_box=hidden_state_0_1)

pred_x_0_1 = Box(
    text=r"$x^{\#1}_{t=0}$",                 
    pos=add_pos(hidden_state_0_1, (0, -1)),
    width=default_width,
    height=0.5)

make_pred_x_0_1 = Arrow(
    start_box=hidden_state_0_1, 
    stop_box=pred_x_0_1)

accuracy_loss_0_1 = Box(
    text=r"$L^{A, \#1}_{t=0}$",                 
    pos=add_pos(pred_x_0_1, (0, -1)),
    width=default_width,
    height=0.5)

make_accuracy_loss_0_1_a = Arrow(
    start_box=pred_x_0_1, 
    stop_box=accuracy_loss_0_1)

make_accuracy_loss_0_1_b = Arrow(
    start_box=real_x_0, 
    stop_box=accuracy_loss_0_1)

complex_loss_0_1 = Box(
    text=r"$L^{C, \#1}_{t=0}$",                 
    pos=add_pos(post_0_1, post_to_complex),
    width=default_width,
    height=0.5)

make_complex_loss_0_1_a = Arrow(
    start_box=prior_0_1, 
    stop_box=complex_loss_0_1)

make_complex_loss_0_1_b = Arrow(
    start_box=post_0_1, 
    stop_box=complex_loss_0_1)



slide_9 = Slide(
    slide_title = 9,
    box_list = [
        do_that_again_side_text,
        hidden_state_m1_1, 
        post_0_1,
        prior_0_1,
        hidden_state_0_1, 
        pred_x_0_1, 
        accuracy_loss_0_1,
        real_x_0,
        complex_loss_0_1],
    arrow_list = [
        make_prior_0_1,
        make_hidden_state_0_1_a, make_hidden_state_0_1_b,
        make_pred_x_0_1,
        make_accuracy_loss_0_1_a, make_accuracy_loss_0_1_b,
        make_complex_loss_0_1_a, make_complex_loss_0_1_b])



# Step 10
post_1_0 = Box(
    text=r"$(\mu^{q,\#0}_{t=1}, \sigma^{q,\#0}_{t=1}) \rightarrow z^{q,\#0}_{t=1}$",  
    pos=add_pos(hidden_state_0_1, hidden_to_post),
    width=prior_post_width,
    height=0.5)

post_1_0_side_text = Box(
    text="Initiate first $t=1$ posterior\ninner state $(\mu = 0, \sigma = 1)$",  
    pos=add_pos(post_1_0, (0, 1)),
    width=0,
    height=0)



slide_10 = Slide(
    slide_title = 10,
    box_list = [
        hidden_state_m1_1, 
        post_0_1,
        prior_0_1,
        hidden_state_0_1, 
        pred_x_0_1, 
        accuracy_loss_0_1,
        real_x_0,
        complex_loss_0_1,
        post_1_0, post_1_0_side_text],
    arrow_list = [
        make_prior_0_1,
        make_hidden_state_0_1_a, make_hidden_state_0_1_b,
        make_pred_x_0_1,
        make_accuracy_loss_0_1_a, make_accuracy_loss_0_1_b,
        make_complex_loss_0_1_a, make_complex_loss_0_1_b])



# Step 11
hidden_state_1_0_side_text = Box(
    text="Generate first\n$t=1$ hidden state",  
    pos=add_pos(hidden_state_1_0, (0, -1)),
    width=0,
    height=0)

make_hidden_state_1_0_a = Arrow(
    start_box=hidden_state_0_1, 
    stop_box=hidden_state_1_0)

make_hidden_state_1_0_b = Arrow(
    start_box=post_1_0, 
    stop_box=hidden_state_1_0)



slide_11 = Slide(
    slide_title = 11,
    box_list = [
        hidden_state_m1_1, 
        post_0_1,
        prior_0_1,
        hidden_state_0_1, 
        pred_x_0_1, 
        accuracy_loss_0_1,
        real_x_0,
        complex_loss_0_1,
        post_1_0, 
        hidden_state_1_0, hidden_state_1_0_side_text],
    arrow_list = [
        make_prior_0_1,
        make_hidden_state_0_1_a, make_hidden_state_0_1_b,
        make_pred_x_0_1,
        make_accuracy_loss_0_1_a, make_accuracy_loss_0_1_b,
        make_complex_loss_0_1_a, make_complex_loss_0_1_b,
        make_hidden_state_1_0_a, make_hidden_state_1_0_b])



# Step 12
do_that_again_side_text = Box(
    text="Perform the previous steps\nfor $t=1$",  
    pos=(2, 3),
    width=0,
    height=0)

prior_1_0 = Box(
    text=r"$(\mu^{p,\#0}_{t=1}, \sigma^{p,\#0}_{t=1}) \rightarrow z^{p,\#0}_{t=1}$",  
    pos=add_pos(hidden_state_0_1, hidden_to_prior),
    width=prior_post_width,
    height=0.5)

make_prior_1_0 = Arrow(
    start_box=hidden_state_0_1, 
    stop_box=prior_1_0)

pred_x_1_0 = Box(
    text=r"$x^{\#0}_{t=1}$",                 
    pos=add_pos(hidden_state_1_0, (0, -1)),
    width=default_width,
    height=0.5)

make_pred_x_1_0 = Arrow(
    start_box=hidden_state_1_0, 
    stop_box=pred_x_1_0)

accuracy_loss_1_0 = Box(
    text=r"$L^{A, \#0}_{t=1}$",                 
    pos=add_pos(pred_x_1_0, (0, -1)),
    width=default_width,
    height=0.5)

real_x_1 = Box(
    text=r"$x_{t=1}$",                 
    pos=add_pos(accuracy_loss_1_0, (0, -1)),
    width=default_width,
    height=0.5)

make_accuracy_loss_1_0_a = Arrow(
    start_box=pred_x_1_0, 
    stop_box=accuracy_loss_1_0)

make_accuracy_loss_1_0_b = Arrow(
    start_box=real_x_1, 
    stop_box=accuracy_loss_1_0)

complex_loss_1_0 = Box(
    text=r"$L^{C, \#0}_{t=1}$",                 
    pos=add_pos(post_1_0, post_to_complex),
    width=default_width,
    height=0.5)

make_complex_loss_1_0_a = Arrow(
    start_box=prior_1_0, 
    stop_box=complex_loss_1_0)

make_complex_loss_1_0_b = Arrow(
    start_box=post_1_0, 
    stop_box=complex_loss_1_0)



slide_12 = Slide(
    slide_title = 12,
    box_list = [
        hidden_state_m1_1, 
        post_0_1,
        prior_0_1,
        hidden_state_0_1, 
        pred_x_0_1, 
        accuracy_loss_0_1,
        real_x_0,
        complex_loss_0_1,
        post_1_0, 
        hidden_state_1_0, 
        do_that_again_side_text,
        prior_1_0,
        hidden_state_1_0, 
        pred_x_1_0, 
        accuracy_loss_1_0,
        real_x_1,
        complex_loss_1_0,
        post_0_1, 
        hidden_state_0_1],
    arrow_list = [
        make_prior_0_1,
        make_hidden_state_0_1_a, make_hidden_state_0_1_b,
        make_pred_x_0_1,
        make_accuracy_loss_0_1_a, make_accuracy_loss_0_1_b,
        make_complex_loss_0_1_a, make_complex_loss_0_1_b,
        make_hidden_state_1_0_a, make_hidden_state_1_0_b,
        make_prior_1_0,
        make_hidden_state_1_0_a, make_hidden_state_1_0_b,
        make_pred_x_1_0,
        make_accuracy_loss_1_0_a, make_accuracy_loss_1_0_b,
        make_complex_loss_1_0_a, make_complex_loss_1_0_b])



# Step 13
second_backprop_side_text = Box(
    text="With backpropogation,\nupdate hidden state $t=-1$ and\nposterior inner state for t = 0, 1",  
    pos=(2, 4),
    width=0,
    height=0)

slide_13 = Slide(
    slide_title = 13,
    box_list = [
        hidden_state_m1_1, 
        post_0_1,
        prior_0_1,
        hidden_state_0_1, 
        pred_x_0_1, 
        accuracy_loss_0_1,
        real_x_0,
        complex_loss_0_1,
        post_1_0, 
        hidden_state_1_0, 
        prior_1_0,
        hidden_state_1_0, 
        pred_x_1_0, 
        accuracy_loss_1_0,
        real_x_1,
        complex_loss_1_0,
        post_0_1, 
        hidden_state_0_1,
        second_backprop_side_text],
    arrow_list = [
        make_prior_0_1,
        make_hidden_state_0_1_a, make_hidden_state_0_1_b,
        make_pred_x_0_1,
        make_accuracy_loss_0_1_a, make_accuracy_loss_0_1_b,
        make_complex_loss_0_1_a, make_complex_loss_0_1_b,
        make_hidden_state_1_0_a, make_hidden_state_1_0_b,
        make_prior_1_0,
        make_hidden_state_1_0_a, make_hidden_state_1_0_b,
        make_pred_x_1_0,
        make_accuracy_loss_1_0_a, make_accuracy_loss_1_0_b,
        make_complex_loss_1_0_a, make_complex_loss_1_0_b])



# Step 14
second_backprop_complete_side_text = Box(
    text="Values updated",  
    pos=(2, 3),
    width=0,
    height=0)

post_0_2 = Box(
    text=r"$(\mu^{q,\#2}_{t=0}, \sigma^{q,\#2}_{t=0}) \rightarrow z^{q,\#2}_{t=0}$",  
    pos=add_pos(hidden_state_m1_2, hidden_to_post),
    width=prior_post_width,
    height=0.5)

post_1_1 = Box(
    text=r"$(\mu^{q,\#1}_{t=1}, \sigma^{q,\#1}_{t=1}) \rightarrow z^{q,\#1}_{t=1}$",  
    pos=add_pos(hidden_state_0_1, hidden_to_post),
    width=prior_post_width,
    height=0.5)



slide_14 = Slide(
    slide_title = 14,
    box_list = [
        second_backprop_complete_side_text,
        hidden_state_m1_2,
        post_0_2,
        post_1_1],
    arrow_list = [])



# Step 15
and_so_on_side_text = Box(
    text="And so on",  
    pos=(2, 3),
    width=0,
    height=0)









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
    text=r"Slide Show for PVRNN Training",                 
    pos=center_pos,
    width=5,
    height=.5)

slide_title = Slide(
    slide_title = "",
    box_list = [title_text],
    arrow_list = [])



intro = Box(
    text="PVRNN remakes previous variables,\nwhich must be considered.\nHow to read variables:",                 
    pos=add_pos(center, (0, 3)),
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

hashtag_side = Box(
    text="",
    pos=add_pos(hashtag_text, (-1.75, 0)),
    width=0,
    height=0)

t_text = Box(
    text="Subscript with $t=$ is\nwhich time-step the\nvariable exists in",
    pos=add_pos(example, (3, -.5)),
    width=0,
    height=0)

t_side = Box(
    text="",
    pos=add_pos(t_text, (-1.25, 0)),
    width=0,
    height=0)

to_hashtag = Arrow(
    start_box=hashtag_side, 
    stop_box=example)

to_t = Arrow(
    start_box=t_side, 
    stop_box=example)

slide_intro = Slide(
    slide_title = "Introduction",
    box_list = [
        center,
        intro,
        example,
        hashtag_text, hashtag_side,
        t_text, t_side],
    arrow_list = [
        to_hashtag,
        to_t])



slide_list = [slide_title, slide_intro] + slide_list



for slide in slide_list:
    slide.plot_slide(min_x_pos, max_x_pos, min_y_pos, max_y_pos, center_pos, axes = False)