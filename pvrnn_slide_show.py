#%% 
from slide_show_tool import *

box_list = []
arrow_list = []

print(1, box_list)

steps_before_first_backprop = 7
steps_before_second_backprop = 15

hidden_to_hidden = (2.5, 0)
hidden_to_prior = (.75, 1.6)
hidden_to_post = (1.75, 1)
post_to_complex = (.25, 1.25)

default_width = .75
prior_post_width = 2.1

# Step 0
hidden_state_m1_0 = Box(
    text=r"$d^{\#0}_{t=-1}$",                 
    pos=(0, 0),
    steps=[i for i in range(0, steps_before_first_backprop)],
    width=default_width,
    height=0.5)
box_list.append(hidden_state_m1_0)

hidden_state_m1_0_side_text = Box(
    text="Initiate hidden state/context/\nlatent state as zeroes",  
    pos=add_pos(hidden_state_m1_0, (2.5, -.1)),
    steps=[0],
    width=3.5,
    height=1)
box_list.append(hidden_state_m1_0_side_text)

post_0_0 = Box(
    text=r"$(\mu^{q,\#0}_{t=0}, \sigma^{q,\#0}_{t=0}) \rightarrow z^{q,\#0}_{t=0}$",  
    pos=add_pos(hidden_state_m1_0, hidden_to_post),
    steps=[i for i in range(0, steps_before_first_backprop)],
    width=prior_post_width,
    height=0.5)
box_list.append(post_0_0)

post_0_0_side_text = Box(
    text="Initiate first $t=0$ posterior\ninner state $(\mu = 0, \sigma = 1)$",  
    pos=add_pos(post_0_0, (3, .1)),
    steps=[0],
    width=3,
    height=1.0)
box_list.append(post_0_0_side_text)


print(2, box_list)



# Step 1
prior_0_0 = Box(
    text=r"$(\mu^{p,\#0}_{t=0}, \sigma^{p,\#0}_{t=0}) \rightarrow z^{p,\#0}_{t=0}$",  
    pos=add_pos(hidden_state_m1_0, hidden_to_prior),
    steps=[i for i in range(1, steps_before_first_backprop)],
    width=prior_post_width,
    height=0.5)
box_list.append(prior_0_0)

prior_0_0_side_text = Box(
    text="Generate first $t=0$\nprior inner state",  
    pos=add_pos(prior_0_0, (0, 1)),
    steps=[1],
    width=3.0,
    height=1.0)
box_list.append(prior_0_0_side_text)

make_prior_0_0 = Arrow(
    start_box=hidden_state_m1_0, 
    stop_box=prior_0_0,
    steps="all")
arrow_list.append(make_prior_0_0)



# Step 2
hidden_state_0_0 = Box(
    text=r"$d^{\#0}_{t=0}$",                 
    pos=add_pos(hidden_state_m1_0, hidden_to_hidden),
    steps=[i for i in range(2, steps_before_first_backprop)],
    width=default_width,
    height=.5)
box_list.append(hidden_state_0_0)

hidden_state_0_0_side_text = Box(
    text="Generate first\n$t=0$ hidden state",  
    pos=add_pos(hidden_state_0_0, (0, -1)),
    steps=[2],
    width=3,
    height=1.0)
box_list.append(hidden_state_0_0_side_text)

make_hidden_state_0_0_a = Arrow(
    start_box=hidden_state_m1_0, 
    stop_box=hidden_state_0_0,
    steps="all")
arrow_list.append(make_hidden_state_0_0_a)

make_hidden_state_0_0_b = Arrow(
    start_box=post_0_0, 
    stop_box=hidden_state_0_0,
    steps="all")
arrow_list.append(make_hidden_state_0_0_b)



# Step 3
pred_x_0_0 = Box(
    text=r"$x^{\#0}_{t=0}$",                 
    pos=add_pos(hidden_state_0_0, (0, -1)),
    steps=[i for i in range(3, steps_before_first_backprop)],
    width=default_width,
    height=0.5)
box_list.append(pred_x_0_0)

pred_x_0_0_side_text = Box(
    text="Generate first estimation of\n$t=0$ observation",  
    pos=add_pos(pred_x_0_0, (0, -1)),
    steps=[3],
    width=3.5,
    height=1.0)
box_list.append(pred_x_0_0_side_text)

make_pred_x_0_0 = Arrow(
    start_box=hidden_state_0_0, 
    stop_box=pred_x_0_0,
    steps="all")
arrow_list.append(make_pred_x_0_0)



# Step 4
accuracy_loss_0_0 = Box(
    text=r"$L^{A, \#0}_{t=0}$",                 
    pos=add_pos(pred_x_0_0, (0, -1)),
    steps=[i for i in range(4, steps_before_first_backprop)],
    width=default_width,
    height=0.5)
box_list.append(accuracy_loss_0_0)

accuracy_loss_0_0_side_text = Box(
    text="First $t=0$ Error, Loss (Accuracy)",  
    pos=(-.25, -2),
    steps=[4],
    width=3.75,
    height=0.5)
box_list.append(accuracy_loss_0_0_side_text)

real_x_0 = Box(
    text=r"$x_{t=0}$",                 
    pos=add_pos(accuracy_loss_0_0, (0, -1)),
    steps=[i for i in range(4, steps_before_first_backprop)],
    width=default_width,
    height=0.5)
box_list.append(real_x_0)

real_x_0_side_text = Box(
    text="Real $t=0$ observation (target)",  
    pos=(-.25, -3),
    steps=[4],
    width=3.75,
    height=0.5)
box_list.append(real_x_0_side_text)

make_accuracy_loss_0_0_a = Arrow(
    start_box=pred_x_0_0, 
    stop_box=accuracy_loss_0_0,
    steps="all")
arrow_list.append(make_accuracy_loss_0_0_a)

make_accuracy_loss_0_0_b = Arrow(
    start_box=real_x_0, 
    stop_box=accuracy_loss_0_0,
    steps="all")
arrow_list.append(make_accuracy_loss_0_0_b)



# Step 5
complex_loss_0_0 = Box(
    text=r"$L^{C, \#0}_{t=0}$",                 
    pos=add_pos(post_0_0, post_to_complex),
    steps=[i for i in range(5, steps_before_first_backprop)],
    width=default_width,
    height=0.5)
box_list.append(complex_loss_0_0)

complexity_loss_0_0_side_text = Box(
    text="First $t=0$ prior/posterior\nDKL, Loss (Complexity)",  
    pos=add_pos(complex_loss_0_0, (0, 1)),
    steps=[5],
    width=3.0,
    height=1)
box_list.append(complexity_loss_0_0_side_text)

make_complexity_loss_0_0_a = Arrow(
    start_box=prior_0_0, 
    stop_box=complex_loss_0_0,
    steps="all")
arrow_list.append(make_complexity_loss_0_0_a)

make_complexity_loss_0_0_b = Arrow(
    start_box=post_0_0, 
    stop_box=complex_loss_0_0,
    steps="all")
arrow_list.append(make_complexity_loss_0_0_b)



# Step 6
first_backprop_side_text = Box(
    text="With backpropogation,\nupdate initial hidden state,\n$t=0$ posterior inner state",  
    pos=(2, 4),
    steps=[6],
    width=3.25,
    height=1)
box_list.append(first_backprop_side_text)



# Step 7
first_backprop_complete_side_text = Box(
    text="Values updated",  
    pos=(2, 3),
    steps=[7],
    width=2,
    height=.5)
box_list.append(first_backprop_complete_side_text)

hidden_state_m1_1 = Box(
    text=r"$d^{\#1}_{t=-1}$",                 
    pos=(0, 0),
    steps=[i for i in range(7, steps_before_second_backprop)],
    width=default_width,
    height=0.5)
box_list.append(hidden_state_m1_1)

post_0_1 = Box(
    text=r"$(\mu^{q,\#1}_{t=0}, \sigma^{q,\#1}_{t=0}) \rightarrow z^{q,\#1}_{t=0}$",  
    pos=add_pos(hidden_state_m1_1, hidden_to_post),
    steps=[i for i in range(7, steps_before_second_backprop)],
    width=prior_post_width,
    height=0.5)
box_list.append(post_0_1)



# Step 8
do_that_again_side_text = Box(
    text="Perform the previous steps\nfor the second time",  
    pos=(2, 3),
    steps=[8],
    width=3.5,
    height=.5)
box_list.append(do_that_again_side_text)

prior_0_1 = Box(
    text=r"$(\mu^{p,\#1}_{t=0}, \sigma^{p,\#1}_{t=0}) \rightarrow z^{p,\#1}_{t=0}$",  
    pos=add_pos(hidden_state_m1_1, hidden_to_prior),
    steps=[i for i in range(8, steps_before_second_backprop)],
    width=prior_post_width,
    height=0.5)
box_list.append(prior_0_1)

make_prior_0_1 = Arrow(
    start_box=hidden_state_m1_1, 
    stop_box=prior_0_1,
    steps="all")
arrow_list.append(make_prior_0_1)

hidden_state_0_1 = Box(
    text=r"$d^{\#1}_{t=0}$",                 
    pos=add_pos(hidden_state_m1_0, hidden_to_hidden),
    steps=[i for i in range(8, steps_before_second_backprop)],
    width=default_width,
    height=0.5)
box_list.append(hidden_state_0_1)

make_hidden_state_0_1_a = Arrow(
    start_box=hidden_state_m1_1, 
    stop_box=hidden_state_0_1,
    steps="all")
arrow_list.append(make_hidden_state_0_1_a)

make_hidden_state_0_1_b = Arrow(
    start_box=post_0_1, 
    stop_box=hidden_state_0_1,
    steps="all")
arrow_list.append(make_hidden_state_0_1_b)

pred_x_0_1 = Box(
    text=r"$x^{\#1}_{t=0}$",                 
    pos=add_pos(hidden_state_0_1, (0, -1)),
    steps=[i for i in range(8, steps_before_second_backprop)],
    width=default_width,
    height=0.5)
box_list.append(pred_x_0_1)

make_pred_x_0_1 = Arrow(
    start_box=hidden_state_0_1, 
    stop_box=pred_x_0_1,
    steps="all")
arrow_list.append(make_pred_x_0_1)

accuracy_loss_0_1 = Box(
    text=r"$L^{A, \#1}_{t=0}$",                 
    pos=add_pos(pred_x_0_1, (0, -1)),
    steps=[i for i in range(8, steps_before_second_backprop)],
    width=default_width,
    height=0.5)
box_list.append(accuracy_loss_0_1)

real_x_0 = Box(
    text=r"$x_{t=0}$",                 
    pos=add_pos(accuracy_loss_0_1, (0, -1)),
    steps=[i for i in range(8, steps_before_second_backprop)],
    width=default_width,
    height=0.5)
box_list.append(real_x_0)

make_accuracy_loss_0_1_a = Arrow(
    start_box=pred_x_0_1, 
    stop_box=accuracy_loss_0_1,
    steps="all")
arrow_list.append(make_accuracy_loss_0_1_a)

make_accuracy_loss_0_1_b = Arrow(
    start_box=real_x_0, 
    stop_box=accuracy_loss_0_1,
    steps="all")
arrow_list.append(make_accuracy_loss_0_1_b)

complex_loss_0_1 = Box(
    text=r"$L^{C, \#1}_{t=0}$",                 
    pos=add_pos(post_0_1, post_to_complex),
    steps=[i for i in range(8, steps_before_second_backprop)],
    width=default_width,
    height=0.5)
box_list.append(complex_loss_0_1)

make_complexity_loss_0_0_a = Arrow(
    start_box=prior_0_1, 
    stop_box=complex_loss_0_1,
    steps="all")
arrow_list.append(make_complexity_loss_0_0_a)

make_complexity_loss_0_0_b = Arrow(
    start_box=post_0_1, 
    stop_box=complex_loss_0_1,
    steps="all")
arrow_list.append(make_complexity_loss_0_0_b)



# Step 9
post_1_0 = Box(
    text=r"$(\mu^{q,\#0}_{t=1}, \sigma^{q,\#0}_{t=1}) \rightarrow z^{q,\#0}_{t=1}$",  
    pos=add_pos(hidden_state_0_1, hidden_to_post),
    steps=[i for i in range(9, steps_before_second_backprop)],
    width=prior_post_width,
    height=0.5)
box_list.append(post_1_0)

post_1_0_side_text = Box(
    text="Initiate first $t=1$ posterior\ninner state $(\mu = 0, \sigma = 1)$",  
    pos=add_pos(post_1_0, (0, 1)),
    steps=[9],
    width=3,
    height=1.0)
box_list.append(post_1_0_side_text)



# Step 10
hidden_state_1_0 = Box(
    text=r"$d^{\#0}_{t=1}$",                 
    pos=add_pos(hidden_state_0_1, hidden_to_hidden),
    steps=[i for i in range(10, steps_before_second_backprop)],
    width=default_width,
    height=0.5)
box_list.append(hidden_state_1_0)

hidden_state_1_0_side_text = Box(
    text="Generate first\n$t=1$ hidden state",  
    pos=add_pos(hidden_state_1_0, (0, -1)),
    steps=[10],
    width=3,
    height=1.0)
box_list.append(hidden_state_1_0_side_text)

make_hidden_state_1_0_a = Arrow(
    start_box=hidden_state_0_1, 
    stop_box=hidden_state_1_0,
    steps="all")
arrow_list.append(make_hidden_state_1_0_a)

make_hidden_state_1_0_b = Arrow(
    start_box=post_1_0, 
    stop_box=hidden_state_1_0,
    steps="all")
arrow_list.append(make_hidden_state_1_0_b)



# Step 11
do_that_again_side_text = Box(
    text="Perform the previous steps\nfor $t=1$",  
    pos=(2, 3),
    steps=[11],
    width=3.5,
    height=.5)
box_list.append(do_that_again_side_text)

prior_1_0 = Box(
    text=r"$(\mu^{p,\#0}_{t=1}, \sigma^{p,\#0}_{t=1}) \rightarrow z^{p,\#0}_{t=1}$",  
    pos=add_pos(hidden_state_0_1, hidden_to_prior),
    steps=[i for i in range(11, steps_before_second_backprop)],
    width=prior_post_width,
    height=0.5)
box_list.append(prior_1_0)

make_prior_1_0 = Arrow(
    start_box=hidden_state_0_1, 
    stop_box=prior_1_0,
    steps="all")
arrow_list.append(make_prior_1_0)

pred_x_1_0 = Box(
    text=r"$x^{\#0}_{t=1}$",                 
    pos=add_pos(hidden_state_1_0, (0, -1)),
    steps=[i for i in range(11, steps_before_second_backprop)],
    width=default_width,
    height=0.5)
box_list.append(pred_x_1_0)

make_pred_x_1_0 = Arrow(
    start_box=hidden_state_1_0, 
    stop_box=pred_x_1_0,
    steps="all")
arrow_list.append(make_pred_x_1_0)

accuracy_loss_1_0 = Box(
    text=r"$L^{A, \#0}_{t=1}$",                 
    pos=add_pos(pred_x_1_0, (0, -1)),
    steps=[i for i in range(11, steps_before_second_backprop)],
    width=0.5,
    height=0.5)
box_list.append(accuracy_loss_1_0)

real_x_1 = Box(
    text=r"$x_{t=1}$",                 
    pos=add_pos(accuracy_loss_1_0, (0, -1)),
    steps=[i for i in range(11, steps_before_second_backprop)],
    width=default_width,
    height=0.5)
box_list.append(real_x_1)

make_accuracy_loss_1_0_a = Arrow(
    start_box=pred_x_1_0, 
    stop_box=accuracy_loss_1_0,
    steps="all")
arrow_list.append(make_accuracy_loss_1_0_a)

make_accuracy_loss_1_0_b = Arrow(
    start_box=real_x_1, 
    stop_box=accuracy_loss_1_0,
    steps="all")
arrow_list.append(make_accuracy_loss_1_0_b)

complex_loss_1_0 = Box(
    text=r"$L^{C, \#0}_{t=1}$",                 
    pos=add_pos(post_1_0, post_to_complex),
    steps=[i for i in range(11, steps_before_second_backprop)],
    width=default_width,
    height=0.5)
box_list.append(complex_loss_1_0)

make_complexity_loss_1_0_a = Arrow(
    start_box=prior_1_0, 
    stop_box=complex_loss_1_0,
    steps="all")
arrow_list.append(make_complexity_loss_1_0_a)

make_complexity_loss_1_0_b = Arrow(
    start_box=post_1_0, 
    stop_box=complex_loss_1_0,
    steps="all")
arrow_list.append(make_complexity_loss_1_0_b)



print(3, box_list)





# Gather minimum and maximum x and y poss
all_x_poss = [box.pos[0] for box in box_list]
all_y_poss = [box.pos[1] for box in box_list]
min_x_pos = min(all_x_poss)
max_x_pos = max(all_x_poss)
min_y_pos = min(all_y_poss)
max_y_pos = max(all_y_poss)



# Gather all slides mentioned by boxes
all_slide_numbers = sorted({s for box in box_list for s in box.steps})



# -------------------------
# MAIN LOOP
# -------------------------
for slide_number in all_slide_numbers:
    plot_slide(slide_number, box_list, arrow_list)