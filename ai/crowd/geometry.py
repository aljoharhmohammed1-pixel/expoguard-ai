"""Normalized polygon geometry, boundary inclusive; no physical coordinates."""
EPS = 1e-12


def cross(a,b,c):
    return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])


def on_segment(p,a,b):
    return abs(cross(a,b,p)) <= EPS and min(a[0],b[0])-EPS <= p[0] <= max(a[0],b[0])+EPS and min(a[1],b[1])-EPS <= p[1] <= max(a[1],b[1])+EPS


def intersects(a,b,c,d):
    turns = cross(a,b,c),cross(a,b,d),cross(c,d,a),cross(c,d,b)
    opposite=lambda x,y:(x>EPS and y<-EPS) or (x<-EPS and y>EPS)
    return (opposite(turns[0],turns[1]) and opposite(turns[2],turns[3])) or any(
        on_segment(p,x,y) for p,x,y in [(a,c,d),(b,c,d),(c,a,b),(d,a,b)])


def validate_polygon(points):
    if len(set(points)) != len(points):
        raise ValueError('ROI vertices must be unique; do not repeat closing vertex')
    edges = list(zip(points,points[1:]+points[:1]))
    for i,b in enumerate(points):
        a,c=points[i-1],points[(i+1)%len(points)]
        if abs(cross(a,b,c))<=EPS and (b[0]-a[0])*(c[0]-b[0])+(b[1]-a[1])*(c[1]-b[1])<0:
            raise ValueError('Adjacent ROI edges cannot overlap')
    if abs(sum(a[0]*b[1]-b[0]*a[1] for a,b in edges)) <= EPS:
        raise ValueError('ROI must have nonzero area')
    for i,(a,b) in enumerate(edges):
        for j,(c,d) in enumerate(edges):
            if j <= i or j == i+1 or (i==0 and j==len(edges)-1):
                continue
            if intersects(a,b,c,d):
                raise ValueError('ROI must be a simple polygon')


def bottom_center(box):
    return ((box[0]+box[2])/2,box[3])


def point_in_polygon(point,polygon):
    inside=False
    x,y=point
    for a,b in zip(polygon,polygon[1:]+polygon[:1]):
        if on_segment(point,a,b):
            return True
        if (a[1]>y)!=(b[1]>y) and x < (b[0]-a[0])*(y-a[1])/(b[1]-a[1])+a[0]:
            inside=not inside
    return inside

