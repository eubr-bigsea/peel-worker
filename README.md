# Peel


Peel is an Algorithmic Explainability Platform, designed as an extension (plugin) to Lemonade. It provides tools to study and evaluate machine learning models with a focus on interpretability and transparency.

Peel is composed of three main services, all containerized with Docker:

* Peel-db – Database service for storing model evaluation and explainability data
* Peel-back – Backend service that exposes Peel’s APIs and integrates with Lemonade
* Peel-worker – Worker service responsible for processing evaluation requests asynchronously

Additionally, Peel requires a running Kafka service for message queuing and task coordination and a Lemonade instance.

## How it Works

1. Users interact with Peel through the Lemonade interface.
2. When a new model evaluation is requested, Lemonade sends a service order that is pushed to a Kafka queue.
3. The Peel-worker consumes the queued request and executes the evaluation process.
4. Results are stored in Peel-db and made available through Peel-back, allowing users to analyze model explainability.

## Getting Started

Clone the docker-lemonade repository:

```
git clone https://github.com/eubr-bigsea/docker-lemonade.git
cd docker-lemonade
git checkout develop
git submodule foreach 'git checkout develop'
git submodule foreach 'git pull develop'
```

Inside docker-lemonade repository, clone both peel repositories:

```
git clone https://github.com/eubr-bigsea/peel-worker.git
git clone https://github.com/eubr-bigsea/peel-back.git
```

Enter in both repositories and create their docker images:

```
docker build -t eubrabigsea/peel-worker:latest . # into peel-worker repo
docker build -t eubrabigsea/peel-back:latest . # into peel-back repo
```

Create the following `docker-compose.yml` file inside docker-lemonade folder:

```
version: '3'
services:

  mysql:
    image: mysql:5.7
    command: mysqld --character-set-server=utf8
      --collation-server=utf8_unicode_ci --init-connect='SET NAMES UTF8;'
      --innodb-flush-log-at-trx-commit=0 --bind-address=0.0.0.0
    environment:
      - MYSQL_ROOT_PASSWORD=lemon
    volumes:
      - ./extras/initdb.d:/docker-entrypoint-initdb.d
      - /srv/lemonade/mysql-dev:/var/lib/mysql
    networks:
      default:
        aliases: [lemonade_db, db]
    restart: on-failure
    ports:
    - 33062:3306

  redis:
    image: redis:4
    networks:
      - default
    restart: on-failure

  thorn:
    build: ./thorn
    image: eubrabigsea/thorn
    volumes:
      - ./config/thorn-config.yaml:/usr/local/thorn/conf/thorn-config.yaml
    networks:
      - default
    restart: unless-stopped

  limonero:
    build: ./limonero
    image: eubrabigsea/limonero
    environment:
      - HADOOP_USER_NAME=hadoop
    volumes:
      - ./config/limonero-config.yaml:/usr/local/limonero/conf/limonero-config.yaml
      - /srv/lemonade-dev/storage:/srv/storage
    networks:
      - default
    restart: unless-stopped

  stand:
    build: ./stand
    image: eubrabigsea/stand
    volumes:
      - ./config/stand-config.yaml:/usr/local/stand/conf/stand-config.yaml
    networks:
      - default
    restart: unless-stopped

  caipirinha:
    build: ./caipirinha
    image: eubrabigsea/caipirinha
    volumes:
      - ./config/caipirinha-config.yaml:/usr/local/caipirinha/conf/caipirinha-config.yaml
    networks:
      - default
    restart: unless-stopped

  tahiti:
    build: ./tahiti
    image: eubrabigsea/tahiti
    volumes:
      - ./config/tahiti-config.yaml:/usr/local/tahiti/conf/tahiti-config.yaml
    networks:
      - default
    restart: unless-stopped

  juicer:
    build: ./juicer
    image: eubrabigsea/juicer
    environment:
      - HADOOP_USER_NAME=hadoop
      - PYSPARK_PYTHON=python3
      - HADOOP_CONF_DIR=/usr/local/juicer/conf
    command: ["/usr/local/juicer/sbin/juicer-daemon.sh", "docker"]
    volumes:
      - ./config/juicer-config.yaml:/usr/local/juicer/conf/juicer-config.yaml
      - /srv/lemonade-dev/storage:/srv/storage
    networks:
      - default
    restart: unless-stopped


  citrus:
    build: ./citrus
    image: eubrabigsea/citrus
    ports:
      - '23456:8080'
    volumes:
      - ./peel-worker/nginx.conf:/etc/nginx/conf.d/default.conf
    networks:
      - default
    restart: unless-stopped

  peel_db:
    image: postgres:15
    container_name: peel_db
    restart: unless-stopped
    environment:
      POSTGRES_USER: myuser
      POSTGRES_PASSWORD: mypassword
      POSTGRES_DB: mydatabase
    ports:
      - "5432:5432"
    volumes:
      - peel-db:/var/lib/postgresql/data/  
    networks:
      - default


  peel_back:
    build: peel-back
    container_name: peel_back
    image: eubrabigsea/peel-back
    restart: unless-stopped
    environment:
      DB_HOST: peel_db
      DB_PORT: 5432
      DB_NAME: mydatabase
      DB_USERNAME: myuser
      DB_PASSWORD: mypassword
      BROKER_LINKS: kafka:9092
      BROKER_TOPIC: peel
    ports:
      - "5001:5000"
    volumes:
      - peel-back:/peel-back/storage
    networks:
      - default

  peel_worker:
    build: peel-worker
    container_name: peel_worker
    restart: unless-stopped
    image: eubrabigsea/peel-worker
    environment:
      DB_HOST: peel_db
      DB_PORT: 5432
      DB_NAME: mydatabase
      DB_USERNAME: myuser
      DB_PASSWORD: mypassword
      BROKER_LINKS: kafka:9092
      BROKER_TOPIC: peel
    ports:
      - "5002:5000"
    volumes:
      - peel-back:/peel-worker/storage
    networks:
      - default

  zookeeper:
    image: confluentinc/cp-zookeeper:7.8.0
    hostname: zookeeper
    container_name: zookeeper
    ports:
      - "2181:2181"
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
      ZOOKEEPER_SERVER_ID: 1
      ZOOKEEPER_SERVERS: zookeeper:2888:3888
    networks:
      - default

  kafka:
    image: confluentinc/cp-kafka:7.8.0
    hostname: kafka
    container_name: kafka
    ports:
      - "9092:9092"
      - "29092:29092"
    networks:
      - default
    depends_on:
      - zookeeper
    environment:
      KAFKA_LISTENERS: PLAINTEXT://0.0.0.0:9092
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://kafka:9092
      KAFKA_ZOOKEEPER_CONNECT: "zookeeper:2181"
      KAFKA_BROKER_ID: 1
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
      KAFKA_TRANSACTION_STATE_LOG_REPLICATION_FACTOR: 1
      KAFKA_TRANSACTION_STATE_LOG_MIN_ISR: 1
      KAFKA_AUTO_CREATE_TOPICS_ENABLE: "true"
      KAFKA_AUTHORIZER_CLASS_NAME: kafka.security.authorizer.AclAuthorizer
      KAFKA_ALLOW_EVERYONE_IF_NO_ACL_FOUND: "true"


volumes:
  peel-db:
  peel-back:
  peel-worker:
          
networks:
  default:


```


Start with: `docker-compose up -d`


Finally:
```
$ docker compose exec -it mysql -u root -plemon
> INSERT INTO thorn.configuration (name, value, enabled, editor, internal) values ("PEEL_HOME","http://localhost:23456/peel", 1, "TEXT", 0);
```



