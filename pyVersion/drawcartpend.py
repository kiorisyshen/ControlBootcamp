import taichi as ti
import math

ground_y = 0.3  # ground y position
wr = 0.012  # wheel radius
rate_pix2rad = 500.0
rate_m2pix = 0.1


def drawcartpend(gui, y, m, M, L):
    """Draw inverted pendulum on cart

    Args:
        gui (taichi gui): gui instance
        y (numpy array): system state [x, x_dot, theta, theta_dot]
        m (float): ball mass
        M (float): cart mass
        L (float): pendulum length
    """
    gui.line(begin=(0.0, ground_y), end=(1.0, ground_y), color=0x0, radius=1)

    hW = 0.1*math.sqrt(M/5)  # cart half width
    hH = 0.05*math.sqrt(M/5)  # cart half height
    mr = 0.02*math.sqrt(m)  # mass radius

    x = y[0, 0]
    theta = y[2, 0]

    cart_center_y = ground_y+wr+wr+hH

    # draw cart and wheels
    gui.circle((x-hW*0.7, ground_y+wr), color=0xffaa77, radius=wr*rate_pix2rad)
    gui.circle((x+hW*0.7, ground_y+wr), color=0xffaa77, radius=wr*rate_pix2rad)
    gui.rect((x-hW, cart_center_y+hH), (x+hW, cart_center_y-hH), color=0x0, radius=1)

    # draw pendulum and ball
    ball_x = x+math.sin(theta)*L*rate_m2pix
    ball_y = cart_center_y+math.cos(theta)*L*rate_m2pix
    gui.circle((ball_x, ball_y), color=0x445566, radius=mr*rate_pix2rad)
    gui.line((x, cart_center_y), (ball_x, ball_y), color=0x0, radius=1)
