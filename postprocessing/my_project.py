def my_project(in_east, in_north, direction):
    '''
    Sample user-defined projection and inverse projection of (lon,lat) to (x,y) 
      
      function [out_east,out_north] = my_project(in_east,in_north,direction) 
      
      DESCRIPTION:
         Define projections between geographical and Euclidean coordinates 
      
      INPUT: 
        in_east   = 1D vector containing longitude (forward) x (reverse)
        in_north  = 1D vector containing latitude  (forward) y (reverse)
        direction = ['forward' ;  'inverse']
                
      OUTPUT:
        (lon,lat) or (x,y) depending on choice of forward or reverse projection
      
      EXAMPLE USAGE
         lon,lat = my_project(x,y,'reverse') 
    '''
    from mpl_toolkits.basemap import pyproj
#    import mpl_toolkits.basemap.pyproj as pyproj
    #state_plane = pyproj.Proj(r'+proj=tmerc +datum=NAD83 +lon_0=-70d10 lat_0=42d50 k=.9999666666666667 x_0=900000 y_0=0 +to_meter=1')
    #state_plane = pyproj.Proj(r'+proj=tmerc +lat_0=42.83333333333334 +lon_0=-70.16666666666667 +k=0.9999666666666667 +x_0=900000 +y_0=0 +ellps=GRS80 +units=m +no_defs')
    state_plane = pyproj.Proj(r'+proj=tmerc +lat_0=42d50 +lon_0=-70d10 +k=0.9999666666666667 +x_0=900000 +y_0=0 +ellps=GRS80 +units=m +no_defs')
    
    wgs = pyproj.Proj(proj='latlong', datum='WGS84', ellps='WGS84')
    if direction=='forward':
        lon = in_east
        lat = in_north
        x,y = pyproj.transform(wgs, state_plane, lon, lat)
        return x, y
    else:
        x = in_east
        y = in_north

        lon, lat = pyproj.transform(state_plane, wgs, x, y)
        return lon, lat
