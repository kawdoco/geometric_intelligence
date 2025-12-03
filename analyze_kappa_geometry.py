import numpy as np, matplotlib.pyplot as plt
from skimage import filters, feature, measure, morphology
from skimage.filters import frangi

# 0) Load your predicted kappa (best: save .npy in your inference script)
# In unet_cpu_infer.py, after computing pred: np.save("outputs_unet/kappa_pred.npy", pred)
K = np.load("outputs_unet/kappa_pred.npy")  # shape HxW

# 1) Quick look + contours
plt.figure(figsize=(6,5))
plt.imshow(K, origin="lower"); plt.contour(K, levels=np.linspace(K.min(), K.max(), 10), colors='k', linewidths=0.5)
plt.title("κ̂ heatmap + contours"); plt.axis('off'); plt.tight_layout()
plt.savefig("outputs_unet/kappa_contours.png", dpi=160); plt.close()

# 2) Peaks (domes/caps)
# Find local maxima on a lightly smoothed map
Ks = filters.gaussian(K, sigma=1.0)
coords = feature.peak_local_max(Ks, min_distance=6, threshold_rel=0.2)
plt.figure(figsize=(6,5)); plt.imshow(K, origin="lower"); plt.scatter(coords[:,1], coords[:,0], s=18, c='r')
plt.title(f"Peaks (found {len(coords)})"); plt.axis('off'); plt.tight_layout()
plt.savefig("outputs_unet/kappa_peaks.png", dpi=160); plt.close()

# 3) Elliptical regions around the top peak
if len(coords) > 0:
    i0, j0 = coords[0]
    thr = np.percentile(K, 85)  # high contour
    mask = K > thr
    lab = measure.label(mask)
    reg = max(measure.regionprops(lab), key=lambda r: r.area)
    y0, x0 = reg.centroid; a = reg.major_axis_length/2; b = reg.minor_axis_length/2; theta = reg.orientation
    ecc = np.sqrt(1 - (b*b)/(a*a + 1e-9))
    print(f"Top region @ ({x0:.1f},{y0:.1f}), axes≈({a:.1f},{b:.1f}), eccentricity≈{ecc:.2f}, angle≈{np.degrees(theta):.1f}°")
    # Draw result
    from matplotlib.patches import Ellipse
    plt.figure(figsize=(6,5)); plt.imshow(K, origin="lower"); 
    ell=Ellipse((x0,y0), width=2*a, height=2*b, angle=np.degrees(-theta), fill=False, lw=2,color='y')
    plt.gca().add_patch(ell)
    plt.title("Elliptical fit to high-κ̂ region"); plt.axis('off'); plt.tight_layout()
    plt.savefig("outputs_unet/kappa_ellipse.png", dpi=160); plt.close()

    # Radial profile around peak
    yy, xx = np.mgrid[0:K.shape[0], 0:K.shape[1]]
    r = np.hypot(xx - x0, yy - y0)
    rmax = min(K.shape)//3
    bins = np.arange(0, rmax, 1.0)
    prof = [K[(r>=rb)&(r<rb+1)].mean() for rb in bins]
    plt.figure(figsize=(5.5,3.0)); plt.plot(bins, prof); plt.xlabel("radius (px)"); plt.ylabel("mean κ̂"); plt.tight_layout()
    plt.savefig("outputs_unet/kappa_radial_profile.png", dpi=160); plt.close()

# 4) Ridges/filaments (Frangi)
Knorm = (K - K.min())/(K.max()-K.min()+1e-12)
ridge = frangi(Knorm, scale_range=(1,3), scale_step=1, beta1=0.5, beta2=15)
plt.figure(figsize=(6,5)); plt.imshow(K, origin="lower"); plt.imshow(ridge, origin="lower", alpha=0.6)
plt.title("Ridge/filament overlay (Frangi)"); plt.axis('off'); plt.tight_layout()
plt.savefig("outputs_unet/kappa_ridges.png", dpi=160); plt.close()

# 5) Streamlines of ∇κ̂ (qualitative flow)
gy, gx = np.gradient(filters.gaussian(K, 1.0))
Y, X = np.mgrid[0:K.shape[0], 0:K.shape[1]]
plt.figure(figsize=(6,5)); plt.imshow(K, origin="lower"); 
plt.streamplot(X, Y, gx, gy, density=1.2, linewidth=0.8, color='w')
plt.title("Streamlines of ∇κ̂ (flow over the surface)"); plt.axis('off'); plt.tight_layout()
plt.savefig("outputs_unet/kappa_streamlines.png", dpi=160); plt.close()

print("Saved: kappa_contours.png, kappa_peaks.png, kappa_ellipse.png, kappa_radial_profile.png, kappa_ridges.png, kappa_streamlines.png")
