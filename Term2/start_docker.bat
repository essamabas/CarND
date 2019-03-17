REM create a new Container with File%
REM docker run -it -p 4567:4567 -v d:/EKF:/EKF udacity/controls_kit:latest
REM start an already container 
docker run --rm -it -p 4567:4567 -v D:\carND\Term2:/Term2 sgalkin/carnd-t2-dev:latest
