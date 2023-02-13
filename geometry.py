import numpy as np
def grad_sphere(field, llon, llat):
    """
    Function to calculate the gradient of a 2D scalar field over a sphere, given the coordinates in degrees on 
    the same 2D grid. The derivatives are taken as second-order differences in the interior, and first-order 
    (forward or backward) on the edges.
    """
    R = 6371.0e3 # Earth radius in km.
    
    field = np.double(field)
    llon = np.double(llon)
    llat = np.double(llat)
    
    costheta = np.cos(llat*np.pi/180)
    
    df_dx = field-field
    df_dx[:,1:-1] = (field[:,2:]-field[:,:-2])/(R*costheta[:,1:-1]*(llon[:,2:]-llon[:,:-2])*np.pi/180)
    df_dx[:,0] = (field[:,1]-field[:,0])/(R*costheta[:,0]*(llon[:,1]-llon[:,0])*np.pi/180)
    df_dx[:,-1] = (field[:,-1]-field[:,-2])/(R*costheta[:,-1]*(llon[:,-1]-llon[:,-2])*np.pi/180)
    
    df_dy = field-field
    df_dy[1:-1,:] = (field[2:,:]-field[:-2,:])/(R*(llat[2:,:]-llat[:-2,:])*np.pi/180)
    df_dy[0,:] = (field[1,:]-field[0,:])/(R*(llat[1,:]-llat[0,:])*np.pi/180)
    df_dy[-1,:] = (field[-1,:]-field[-2,:])/(R*(llat[-1,:]-llat[-2,:])*np.pi/180)
    
    return df_dx, df_dy

def div_sphere(field_a, field_b, llon, llat):
    """
    Function to calculate the divergence of a 2D vectorial field over a sphere, given the coordinates in degrees on 
    the same 2D grid. The derivatives are taken as second-order differences in the interior, and first-order 
    (forward or backward) on the edges.
    """
    R = 6371.0e3 # Earth radius in km.
    
    field_a = np.double(field_a)
    field_b = np.double(field_b)
    llon = np.double(llon)
    llat = np.double(llat)
    
    costheta = np.cos(llat*np.pi/180)

    div_a = field_a-field_a
    div_a[:,1:-1] = (field_a[:,2:]-field_a[:,:-2])/(R*costheta[:,1:-1]*(llon[:,2:]-llon[:,:-2])*np.pi/180)
    div_a[:,0] = (field_a[:,1]-field_a[:,0])/(R*costheta[:,0]*(llon[:,1]-llon[:,0])*np.pi/180)
    div_a[:,-1] = (field_a[:,-1]-field_a[:,-2])/(R*costheta[:,-1]*(llon[:,-1]-llon[:,-2])*np.pi/180)
    
    div_b = field_b-field_b
    div_b[1:-1,:] = (field_b[2:,:]*costheta[2:,:]-field_b[:-2,:]*costheta[:-2,:])/(R*costheta[1:-1,:]*(llat[2:,:]-llat[:-2,:])*np.pi/180)
    div_b[0,:] = (field_b[1,:]*costheta[1,:]-field_b[0,:]*costheta[0,:])/(R*costheta[0,:]*(llat[1,:]-llat[0,:])*np.pi/180)
    div_b[-1,:] = (field_b[-1,:]*costheta[-1,:]-field_b[-2,:]*costheta[-2,:])/(R*costheta[-1,:]*(llat[-1,:]-llat[-2,:])*np.pi/180)
        
    div = div_a + div_b
    return div
  
def nan_gaussian_filter(field,sigma):
    """
    Function to smooth the field ignoring the NaNs.
    I follow the first answer here 
    https://stackoverflow.com/questions/18697532/gaussian-filtering-a-image-with-nan-in-python
    By default, the filter is truncated at 4 sigmas.
    """
    from scipy.ndimage import gaussian_filter
    
    field = np.double(field)
    
    # Take the original field and replace the NaNs with zeros.
    field0 = field.copy()
    field0[np.isnan(field)] = 0
    ff = gaussian_filter(field0, sigma=sigma)
    
    # Create the smoothed weight field.
    weight = 0*field.copy()+1
    weight[np.isnan(field)] = 0
    ww = gaussian_filter(weight, sigma=sigma)
    
    zz = ff/(ww*weight) # This rescale for the actual weights used in the filter and set to NaN where the field
                        # was originally NaN.
    return zz
  
def crop_field(field,lon,lat,area):
    """
    Function to crop a 2d field over the area specified by the list
    area = [minlon,maxlon,minlat,maxlat]
    """
    valid_indices = np.argwhere((lon>area[0])&(lon<area[1])&(lat>area[2])&(lat<area[3]))
    x_min, y_min = valid_indices.min(axis=0)
    x_max, y_max = valid_indices.max(axis=0)
    return field[x_min:x_max+1, y_min:y_max+1]
