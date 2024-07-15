video:
	manim -pql amos-seminar/main.py Seminar
	python -m manim_slides convert Seminar slides/presentation.html
	make wip

wip:
	git add .
	git commit -m "Auto commit."
	git push
