Welcome to the Jupyter Guide to Linear Algebra
============================


The Jupyter Guide to Linear Algebra covers many of the core topics that would appear in an introductory course on linear algebra, together with several applications.  The guide also provides a brief introduction to the Python programming langauge, with focus on the portions that are relevant to linear algebra computations, as well as some general guidance on programming.  In its current form, the Jupyter Guide to Linear Algebra is not intended to be a replacement for a textbook in a traditional university course, although it should prove useful to students in such courses.  The guide may also be useful to those who have some knowledge of linear algebra, and wish to learn how to carry out computations in Python, or experienced Jupyter users that would like to learn a bit about Linear Algebra.

Features:

- Development of a module to be used along side of the Jupyter Guide to Linear Algebra, or independently.
- Exercises aimed at exploring linear algebra concepts, as well as exercises to practice writing Python code.
- Instruction on the basic use of NumPy, SciPy, and Matplotlib.

The code supplied in Jupyter Guide to Linear Algebra performs all operations numerically.  We do not make use of SymPy to perform symbolic computations, which hide the practical challenges of accuracy and stability.  We will discover and acknowledge roundoff error very early in the guide, but we will not provide a detailed error analysis, nor advanced algorithms to minimize its impact.  We will also not be overly concerned with computational efficiency.  This resource is meant for a first course in linear algebra, and we will leave the larger challenges of numerical linear algebra for a second course.   

The Jupyter Guide to Linear Algebra presents the mathematics in a relatively informal way.  We offer explanations in place of proofs and do not follow a traditional model of definitions and theorems.  Instead our main objective is to present methods to solve problems, demonstrate how to carry out calculations, provide the basic terminology of linear algebra, and examine ways in which the abstract ideas can be used in practical ways.

### How to use this Guide

The real purpose of this material is for the reader to engage with the material by experimenting and trying out computations for themselves.  The Jupyter Guide to Linear Algebra is currently distributed in two forms, either a Jupyter book, or a collection of Jupyter notebooks.  The Jupyter book version may be found as a pdf or website.  If you are reading a pdf version, please find the digital version at [bvanderlei.github.io/jupyter-guide-to-linear-algebra/intro.html](https://bvanderlei.github.io/jupyter-guide-to-linear-algebra/intro.html).


If you are reading this on a website, you can launch the notebook for the current section in BinderHub by clicking the rocket icon in the upper right corner of the page.  This will create an interactive Jupyter session which may take a minute to load.  Once it is complete you can edit the notebook, run your own code, and download your work when you are finished.  If you are an experinced Jupyter user, you may also choose to download the notebooks directly.  The complete collection of notebooks can be found at [github.com/bvanderlei/jupyter-guide-to-linear-algebra](https://github.com/bvanderlei/jupyter-guide-to-linear-algebra).      

### Prerequisites or Corequisites

An introductory programming course, or an equivalent programming experience would be useful to the reader.  While we do introduce all of the syntax needed to write Python scripts, we do not delve into the details of traditional topics in an introductory programming course (data types, logic, iteration, and complex data structures).  We address the features of Python that we use most frequetly, and provide only brief mentions of those that are tangential to our goals.

Some applications make use of Calculus and will be noted.  Full appreciation of these sections will require an introductory Calculus course.