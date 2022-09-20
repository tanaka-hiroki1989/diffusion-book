import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot(h,m,c):
    fig = plt.figure(figsize=(5,5))
    ax = plt.axes()
    cir1 = patches.Circle(xy=(0, 0), radius=1.0, fill = False)
    cir2 = patches.Circle(xy=(0, 0), radius=2.0, fill = False)
    cir3 = patches.Circle(xy=(0, 0), radius=3.0, fill = False)
    colors = ["blue" if x_c==0 else "red" for x_c in c]
    ax.set_xlim(-5.0,5.0)
    ax.set_ylim(-5.0,5.0)
    ax.scatter(h.T[0], h.T[1], s=1.0, color=colors,alpha=0.5)
    ax.add_patch(cir1)
    ax.add_patch(cir2)
    ax.add_patch(cir3)
    ax.set_aspect('equal')
    plt.savefig("fig/"+str(m))