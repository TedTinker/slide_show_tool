#%% 
#%%
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math



def add_pos(box, pos_2):
    pos_1 = box.pos
    return(pos_1[0] + pos_2[0], pos_1[1] + pos_2[1])
    
    

class Box:
    def __init__(self, text, pos, width=1.0, height=1.0):
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
        self.width = width
        self.height = height
        
        

class Arrow:
    def __init__(
        self,
        start_box,
        stop_box,
        start_pos=(0, 0),
        stop_pos=(0, 0),
        double_arrow=False,
        color="black",
        text=None  
    ):
        """
        start_box, stop_box: Box objects.
        start_pos, stop_pos: (dx, dy) offsets to shift the arrow start/end.
        double_arrow: whether it's a <-> or -> arrow.
        color: arrow color (default black).
        text: optional text label to place at the arrow's midpoint.
        """
        self.start_box = start_box
        self.stop_box = stop_box
        self.start_pos = start_pos
        self.stop_pos = stop_pos
        self.double_arrow = double_arrow
        self.color = color
        self.text = text  

        
        

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



class Slide:
    def __init__(self, slide_title = "TITLE", box_list = [], arrow_list = [], side_text_list = []):
        self.slide_title = slide_title
        self.box_list = box_list 
        self.arrow_list = arrow_list
        self.side_text_list = side_text_list
        
    def plot_slide(self, min_x_pos, max_x_pos, min_y_pos, max_y_pos, center_pos, axes = False):
        fig, ax = plt.subplots(figsize=(15, 15))  # Larger figure_
        ax.set_title(self.slide_title, fontsize=16)

        # Draw each box as a rectangle + text inside
        for box in self.box_list + self.side_text_list:
            if(box.pos == "center"):
                (cx, cy) = center_pos 
            else:
                (cx, cy) = box.pos
            # The lower-left corner of the rectangle
            llx = cx - box.width/2.0
            lly = cy - box.height/2.0
            
            # Draw rectangle
            rect = patches.Rectangle(
                (llx, lly),
                box.width,
                box.height,
                edgecolor="black",
                facecolor="white"
            )
            ax.add_patch(rect)
            
            # If user manually included $ in box.text, they want math mode.
            # Otherwise, show it as plain text.
            if "$" in box.text:
                label = box.text  # the user is explicitly in math mode
            else:
                label = box.text  # plain text

            ax.text(cx, cy, label,
                    ha="center", va="center", fontsize=20)
        
        # Draw arrows
        for arrow in self.arrow_list:
            # Only draw arrow if both boxes are on this slide
            if (arrow.start_box in self.box_list and
                arrow.stop_box in self.box_list):

                # Find the default anchor points on each box's edge
                start_x, start_y = box_edge_anchor(arrow.start_box, arrow.stop_box)
                end_x, end_y = box_edge_anchor(arrow.stop_box, arrow.start_box)

                # Apply any user-defined offsets
                start_x += arrow.start_pos[0]
                start_y += arrow.start_pos[1]
                end_x   += arrow.stop_pos[0]
                end_y   += arrow.stop_pos[1]

                style = "<->" if arrow.double_arrow else "->"
                
                ax.annotate(
                    "",  # no label text
                    xy=(end_x, end_y),
                    xytext=(start_x, start_y),
                    arrowprops=dict(
                        arrowstyle=style,
                        color=arrow.color,
                        linewidth=1.5
                    )
                )
                mid_x = (start_x + end_x) / 2.0
                mid_y = (start_y + end_y) / 2.0

                if arrow.text is not None:
                    if "$" in arrow.text:
                        label = arrow.text
                    else:
                        label = arrow.text  
                    
                    ax.text(mid_x, mid_y, label,
                            ha="center", va="center", 
                            color=arrow.color,
                            fontsize=16,
                            bbox=dict(facecolor="white", edgecolor="none", pad=0.0))
            else:
                print(f"Warning: arrow from {arrow.start_box.text} to "
                    f"{arrow.stop_box.text} in a slide without those boxes.")

        
        ax.set_xlim(min_x_pos - 1, max_x_pos + 1)
        ax.set_ylim(min_y_pos - 1, max_y_pos + 1)
        ax.set_aspect('equal', adjustable='box')
        if(not axes):
            plt.axis('off')
        plt.savefig(f"saved_slides/slide_{self.slide_title}.png", dpi=150)
        plt.show()
        plt.close()



def get_sizes(slide_list):
    min_xs, max_xs, min_ys, max_ys = [], [], [], []
    for slide in slide_list:
        # Gather minimum and maximum x and y poss
        all_x_poss = [box.pos[0] for box in slide.box_list if box.pos != "center"]
        all_y_poss = [box.pos[1] for box in slide.box_list if box.pos != "center"]
        if(all_x_poss != []):
            min_xs.append(min(all_x_poss))
            max_xs.append(max(all_x_poss))
        if(all_y_poss != []):
            min_ys.append(min(all_y_poss))
            max_ys.append(max(all_y_poss))
    min_x_pos = min(min_xs)
    max_x_pos = max(max_xs)
    min_y_pos = min(min_ys)
    max_y_pos = max(max_ys)
    
    center_pos = ((max_x_pos + min_x_pos) / 2, (max_y_pos + min_y_pos) / 2)
    
    return min_x_pos, max_x_pos, min_y_pos, max_y_pos, center_pos



if(__name__ == "__main__"):
    # -------------------------
    # Example Data
    # -------------------------
    box_list = []
    arrow_list = []

    a = Box(
        text=r"$A$",                 
        pos=(0, 0),
        width=1,
        height=1)
    box_list.append(a)

    b = Box(
        text=r"$B$",                 
        pos=(2, 0),
        width=1,
        height=1)
    box_list.append(b)

    c = Box(
        text=r"$C$",                 
        pos=(0, 2),
        width=1,
        height=1)
    box_list.append(c)

    d = Box(
        text=r"$D$",                 
        pos=(2, 4),
        width=1,
        height=1)
    box_list.append(d)

    arrow_1 = Arrow(
        start_box=a, 
        stop_box=b,
        text = "$LaTeX$\ncapable")
    arrow_list.append(arrow_1)

    arrow_2 = Arrow(
        start_box=c, 
        stop_box=d,
        color = "red")
    arrow_list.append(arrow_2)

    arrow_3 = Arrow(
        start_box=a, 
        stop_box=d)
    arrow_list.append(arrow_3)
    
    
    slide_list = [
        Slide(
            slide_title = 1,
            box_list = [a],
            arrow_list = []),
        
        Slide(
            slide_title = 2,
            box_list = [a, b],
            arrow_list = [arrow_1]),
        
        Slide(
            slide_title = 3,
            box_list = [a, b, c],
            arrow_list = [arrow_1]),
        
        Slide(
            slide_title = 4,
            box_list = [a, b, c, d],
            arrow_list = [arrow_1, arrow_2, arrow_3]),
    ]
    
    min_x_pos, max_x_pos, min_y_pos, max_y_pos, center_pos = get_sizes(slide_list)
    for slide in slide_list:
        slide.plot_slide(min_x_pos, max_x_pos, min_y_pos, max_y_pos, center_pos)