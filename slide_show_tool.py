#%% 
#%%
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math

def add_pos(box, pos_2):
    pos_1 = box.pos
    return(pos_1[0] + pos_2[0], pos_1[1] + pos_2[1])
    

class Box:
    def __init__(self, text, pos, steps, width=1.0, height=1.0):
        """
        text: String representing the label for this box.
              - If you want math mode, include the $...$ in text yourself.
              - Otherwise, use plain text with no $.
        pos: (x, y) center of the box in data coordinates
        steps: list of slide numbers on which this box should appear
        width, height: approximate box dimensions in data coords
        """
        self.text = text
        self.pos = pos
        self.steps = steps
        self.width = width
        self.height = height

class Arrow:
    def __init__(self, start_box, stop_box, steps="all", double_arrow=False, color="black"):
        """
        start_box, stop_box: Box objects
        steps: "all" or list of slide indices on which this arrow should appear
        double_arrow: whether it's a <-> or -> arrow
        color: arrow color
        """
        self.start_box = start_box
        self.stop_box = stop_box
        self.steps = steps
        self.double_arrow = double_arrow
        self.color = color

def box_edge_anchor(boxA, boxB):
    """
    Return (xA, yA), the point on the edge of boxA's rectangle that 
    lies on the line from boxA center to boxB center.
    """
    (xA, yA) = boxA.pos
    (xB, yB) = boxB.pos
    
    # Half-sizes of the box
    half_w = boxA.width / 2.0
    half_h = boxA.height / 2.0
    
    dx = xB - xA
    dy = yB - yA
    angle = math.atan2(dy, dx)
    
    # For a rectangle aligned with the axes, the intersection with 
    # the box boundary in direction angle can be computed using:
    #    x^2 / half_w^2 + y^2 / half_h^2 = 1
    # param = 1 / sqrt( (cos(theta)/half_w)^2 + (sin(theta)/half_h)^2 )
    c = math.cos(angle)
    s = math.sin(angle)
    
    # Avoid zero for degenerate boxes
    if half_w <= 0.0: 
        half_w = 0.1
    if half_h <= 0.0:
        half_h = 0.1

    param = 1.0 / math.sqrt((c/half_w)**2 + (s/half_h)**2)
    
    edge_x = xA + param * c
    edge_y = yA + param * s
    return (edge_x, edge_y)





# -------------------------
# PLOTTING FUNCTION
# -------------------------
def plot_slide(slide_number, box_list, arrow_list):
    """
    Plots all boxes and arrows that are relevant for the given slide_number.
    """
    
    # Gather minimum and maximum x and y poss
    all_x_poss = [box.pos[0] for box in box_list]
    all_y_poss = [box.pos[1] for box in box_list]
    min_x_pos = min(all_x_poss)
    max_x_pos = max(all_x_poss)
    min_y_pos = min(all_y_poss)
    max_y_pos = max(all_y_poss)

    # Gather all slides mentioned by boxes
    all_slide_numbers = sorted({s for box in box_list for s in box.steps})
    fig, ax = plt.subplots(figsize=(15, 15))  # Larger figure
    ax.set_title(f"Slide {slide_number}", fontsize=16)
    
    # Boxes to display on this slide
    boxes_on_this_slide = [b for b in box_list if slide_number in b.steps]
    
    # Draw each box as a rectangle + text inside
    for b in boxes_on_this_slide:
        (cx, cy) = b.pos
        # The lower-left corner of the rectangle
        llx = cx - b.width/2.0
        lly = cy - b.height/2.0
        
        # Draw rectangle
        rect = patches.Rectangle(
            (llx, lly),
            b.width,
            b.height,
            edgecolor="black",
            facecolor="white"
        )
        ax.add_patch(rect)
        
        # If user manually included $ in b.text, they want math mode.
        # Otherwise, show it as plain text.
        if "$" in b.text:
            label = b.text  # the user is explicitly in math mode
        else:
            label = b.text  # plain text

        ax.text(cx, cy, label,
                ha="center", va="center", fontsize=20)
    
    # Draw arrows
    for arr in arrow_list:
        if arr.steps == "all" or slide_number in arr.steps:
            # Check both boxes are on this slide
            if (arr.start_box in boxes_on_this_slide and 
                arr.stop_box in boxes_on_this_slide):
                
                start_x, start_y = box_edge_anchor(arr.start_box, arr.stop_box)
                end_x, end_y = box_edge_anchor(arr.stop_box, arr.start_box)
                
                # Arrow style
                style = "<->" if arr.double_arrow else "->"
                
                ax.annotate(
                    "",  # no text
                    xy=(end_x, end_y),
                    xytext=(start_x, start_y),
                    arrowprops=dict(
                        arrowstyle=style, 
                        color=arr.color, 
                        linewidth=1.5
                    )
                )
    
    ax.set_xlim(min_x_pos - 1, max_x_pos + 1)
    ax.set_ylim(min_y_pos - 1, max_y_pos + 1)
    ax.set_aspect('equal', adjustable='box')
    plt.show()






if(__name__ == "__main__"):
    # -------------------------
    # Example Data
    # -------------------------
    box_list = []
    arrow_list = []

    a = Box(
        text=r"$A$",                 
        pos=(0, 0),
        steps=[i for i in range(0, 4)],
        width=1,
        height=1)
    box_list.append(a)

    b = Box(
        text=r"$B$",                 
        pos=(2, 0),
        steps=[i for i in range(1, 4)],
        width=1,
        height=1)
    box_list.append(b)

    c = Box(
        text=r"$C$",                 
        pos=(0, 2),
        steps=[i for i in range(2, 4)],
        width=1,
        height=1)
    box_list.append(c)

    d = Box(
        text=r"$D$",                 
        pos=(2, 2),
        steps=[i for i in range(3, 4)],
        width=1,
        height=1)
    box_list.append(d)

    arrow_1 = Arrow(
        start_box=a, 
        stop_box=b,
        steps="all")
    arrow_list.append(arrow_1)

    arrow_2 = Arrow(
        start_box=c, 
        stop_box=d,
        steps="all")
    arrow_list.append(arrow_2)

    arrow_3 = Arrow(
        start_box=a, 
        stop_box=d,
        steps="all")
    arrow_list.append(arrow_3)
    
    # -------------------------
    # MAIN LOOP
    # -------------------------
    for slide_number in [0, 1, 2, 3]:
        plot_slide(slide_number, box_list, arrow_list)
