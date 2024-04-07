# JET

Numerical solution of a 2D turbulent hot jet exiting from a nozzle into still atmosphere.

The numerical solution procedure adopts Keller's BOX scheme. It is slightly changed, so that linearizations are skipped and instead a non-linear solver ([fsolve](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fsolve.html)) from the [SciPy package](https://docs.scipy.org/doc/scipy/reference/index.html) is adopted.

During my master thesis (1986) I had to rework the numerical solution procedure and also the software given in the book "Physical and computational aspects of convective heat transfer" by T. Cebeci and P. Bradshaw.

The code was done in FORTRAN 77 and the plotting was performed using the BENSON plot library on a VT200 or VT300 series monochrome terminal with graphics capability. The computer was a MicroVAX II running the VMS operating system.

This work here is revisiting the topic with modern programming tools like Python and Matplotlib. I reworked the software code from my thesis mainly for sentimental reasons.
