FROM debian
RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests --assume-yes postfix && rm -rf /var/*/apt/

COPY main.cf /etc/postfix/main.cf
RUN sed -i 's/^smtp .* smtpd$/smtp inet n - n - - smtpd/' /etc/postfix/master.cf

ENTRYPOINT [ "postfix", "start-fg" ]
