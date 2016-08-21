def PCA(image1)

    #convert to gray
    hlp = np.copy(image1)
    image_gray = hlp/np.nanmax(image1) * 255.0   
    data = image_gray
    
    