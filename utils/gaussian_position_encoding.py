import numpy as np

def gaussian_state_embedding(joint_states, num_features=10000, s=0.1):
    """
    The angle of each robot joint are represented into a polar coordinate point as 
    (x=cos(angle), y=sin(angle)). Then, the 2D plane is converted into a 2D gaussian 
    distribution to have mean, stddev as below.
    mean = (x, y)
    stddev = 
      if
        vel == 0: sigma_x=sigma_y=c (c: constant)
        vel  > 0: sigma_x += atan(vel)    -> sigma_x > sigma_y
        vel  < 0: sigma_y -= atna(vel)    -> sigma_x < sigma_y
    joint_states: [7*2] [angle, velocity]
    num_features: number of features for positional encoding
    s: default sigma
    """    
    x = np.linspace(-1, 1, np.sqrt(num_features).astype(np.uint8))
    y = np.linspace(-1, 1, np.sqrt(num_features).astype(np.uint8))
    xx, yy = np.meshgrid(x, y) # create x, y variable for 2d space
    state_embeddings = np.zeros((len(joint_states), num_features))
    for i in range(len(joint_states)):
        angle, velocity = joint_states[i]
        mean_x, mean_y = np.cos(angle), np.sin(angle)  # polar coordinate concept
        sigma_x = sigma_y = s
        if velocity > 0:
            sigma_x += np.arctan(velocity)/2  # sigma_x > sigma_y
        else:
            sigma_y += -np.arctan(velocity)/2  # sigma_x < sigma_y
        z = np.exp( -( (xx-mean_x)**2/(2*sigma_x**2)+(yy-mean_y)**2/(2*sigma_y**2) ) )  # Gaussian distribution in 2D        
        state_embeddings[i][:] = z.flatten()
    return state_embeddings # [7, 10000]