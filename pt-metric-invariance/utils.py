import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import os
from sklearn.manifold import TSNE


def do_plot(x_train, y_train, model, device, plot_dir, name): 
    x_train = x_train.to(device)

    # print("plot-"+name)
    # import pdb; pdb.set_trace()
    
    print(name)
    train_outputs = model.get_embedding(x_train)

    tsne = TSNE(random_state=0)
    train_tsne_embeds = tsne.fit_transform(train_outputs.cpu().detach().numpy())
    scatter(train_tsne_embeds, y_train.cpu().numpy(), root=plot_dir, subtitle=f'{name} TNN distribution')
    
def tsne_plot(model, device, viz_train_loader, viz_test_loader, mdir, iter_idx):
    print("*****Plotting embeddings at iter: %s****" % iter_idx)
    plot_dir = os.path.join(mdir, 'plots')

    x_train, y_train, x_train_invar = next(iter(viz_train_loader))
    x_test, y_test, x_test_invar = next(iter(viz_test_loader))

    import pdb; pdb.set_trace();


    # x_train_invar = x_train_invar.unsqueeze(1)
    # x_test_invar = x_test_invar.unsqueeze(1)

    do_plot(x_train, y_train, model, device, plot_dir, "Clean Train")
    do_plot(x_test, y_test, model, device, plot_dir, "Clean Test")
    do_plot(x_train_invar, y_train, model, device, plot_dir, "Invariance Train")
    do_plot(x_test_invar, y_test, model, device, plot_dir, "Invariance Test")


    


def scatter(x, labels, root='plot', subtitle=None, dataset='MNIST'):
    mnist_classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                '#bcbd22', '#17becf']
    num_classes = len(set(labels)) # Calculate the number of classes
    palette = np.array(sns.color_palette("hls", num_classes)) # Choosing color

    ## Create a seaborn scatter plot ##
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')

    for i in range(10):
        inds = np.where(labels==i)[0]
        plt.scatter(x[inds,0], x[inds,1], lw=0, s=40, alpha=0.5, color=colors[i])

    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    plt.legend(mnist_classes)    
    # sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,
    #                 c=palette[labels.astype(np.int)])


    ax.axis('off')
    ax.axis('tight')
    ## ---------------------------- ##
    
    # Add label on top of each cluster ##
    txts = []
    for i in range(num_classes):
        # Position of each label.
        xtext, ytext = np.median(x[labels == i, :], axis=0)
        txt = ax.text(xtext, ytext, mnist_classes[i], fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)        
        
    ## ---------------------------- ##    
        
    if subtitle != None:
        plt.suptitle(subtitle)
        
    if not os.path.exists(root):
        os.makedirs(root)    

    plt.savefig(os.path.join(root, str(subtitle)))

def fetch_n_channels(dataloader): 
    x_train, _, _ = next(iter(dataloader))
    return x_train.shape[1]


# import matplotlib
# import matplotlib.pyplot as plt

# mnist_classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
# colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
#               '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
#               '#bcbd22', '#17becf']

# def plot_embeddings(embeddings, targets, xlim=None, ylim=None):
#     plt.figure(figsize=(10,10))
#     for i in range(10):
#         inds = np.where(targets==i)[0]
#         plt.scatter(embeddings[inds,0], embeddings[inds,1], alpha=0.5, color=colors[i])
#     if xlim:
#         plt.xlim(xlim[0], xlim[1])
#     if ylim:
#         plt.ylim(ylim[0], ylim[1])
#     plt.legend(mnist_classes)

# def extract_embeddings(dataloader, model):
#     with torch.no_grad():
#         model.eval()
#         embeddings = np.zeros((len(dataloader.dataset), 2))
#         labels = np.zeros(len(dataloader.dataset))
#         k = 0
#         for images, target in dataloader:
#             if cuda:
#                 images = images.cuda()
#             embeddings[k:k+len(images)] = model.get_embedding(images).data.cpu().numpy()
#             labels[k:k+len(images)] = target.numpy()
#             k += len(images)
#     return embeddings, labels