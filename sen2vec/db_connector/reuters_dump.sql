--
-- PostgreSQL database dump
--

-- Dumped from database version 9.5.4
-- Dumped by pg_dump version 9.5.4

SET statement_timeout = 0;
SET lock_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SET check_function_bodies = false;
SET client_min_messages = warning;
SET row_security = off;

SET search_path = public, pg_catalog;

SET default_tablespace = '';

SET default_with_oids = false;

--
-- Name: topic; Type: TABLE; Schema: public; Owner: naeemul
--

CREATE TABLE topic (
    id integer NOT NULL,
    name text,
    category text
);


ALTER TABLE topic OWNER TO naeemul;

--
-- Name: Topic_id_seq; Type: SEQUENCE; Schema: public; Owner: naeemul
--

CREATE SEQUENCE "Topic_id_seq"
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE "Topic_id_seq" OWNER TO naeemul;

--
-- Name: Topic_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: naeemul
--

ALTER SEQUENCE "Topic_id_seq" OWNED BY topic.id;


--
-- Name: document; Type: TABLE; Schema: public; Owner: naeemul
--

CREATE TABLE document (
    id integer NOT NULL,
    title text,
    content text,
    filename text,
    metadata text
);


ALTER TABLE document OWNER TO naeemul;

--
-- Name: document_paragraph; Type: TABLE; Schema: public; Owner: naeemul
--

CREATE TABLE document_paragraph (
    document_id integer,
    paragraph_id integer,
    "position" integer
);


ALTER TABLE document_paragraph OWNER TO naeemul;

--
-- Name: document_topic; Type: TABLE; Schema: public; Owner: naeemul
--

CREATE TABLE document_topic (
    document_id integer,
    topic_id integer
);


ALTER TABLE document_topic OWNER TO naeemul;

--
-- Name: paragraph; Type: TABLE; Schema: public; Owner: naeemul
--

CREATE TABLE paragraph (
    id integer NOT NULL,
    content text
);


ALTER TABLE paragraph OWNER TO naeemul;

--
-- Name: paragraph_id_seq; Type: SEQUENCE; Schema: public; Owner: naeemul
--

CREATE SEQUENCE paragraph_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE paragraph_id_seq OWNER TO naeemul;

--
-- Name: paragraph_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: naeemul
--

ALTER SEQUENCE paragraph_id_seq OWNED BY paragraph.id;


--
-- Name: paragraph_sentence; Type: TABLE; Schema: public; Owner: naeemul
--

CREATE TABLE paragraph_sentence (
    paragraph_id integer,
    sentence_id integer,
    "position" integer
);


ALTER TABLE paragraph_sentence OWNER TO naeemul;

--
-- Name: sentence; Type: TABLE; Schema: public; Owner: naeemul
--

CREATE TABLE sentence (
    id integer NOT NULL,
    content text
);


ALTER TABLE sentence OWNER TO naeemul;

--
-- Name: sentence_id_seq; Type: SEQUENCE; Schema: public; Owner: naeemul
--

CREATE SEQUENCE sentence_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE sentence_id_seq OWNER TO naeemul;

--
-- Name: sentence_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: naeemul
--

ALTER SEQUENCE sentence_id_seq OWNED BY sentence.id;


--
-- Name: id; Type: DEFAULT; Schema: public; Owner: naeemul
--

ALTER TABLE ONLY paragraph ALTER COLUMN id SET DEFAULT nextval('paragraph_id_seq'::regclass);


--
-- Name: id; Type: DEFAULT; Schema: public; Owner: naeemul
--

ALTER TABLE ONLY sentence ALTER COLUMN id SET DEFAULT nextval('sentence_id_seq'::regclass);


--
-- Name: id; Type: DEFAULT; Schema: public; Owner: naeemul
--

ALTER TABLE ONLY topic ALTER COLUMN id SET DEFAULT nextval('"Topic_id_seq"'::regclass);


--
-- Name: Document_pkey; Type: CONSTRAINT; Schema: public; Owner: naeemul
--

ALTER TABLE ONLY document
    ADD CONSTRAINT "Document_pkey" PRIMARY KEY (id);


--
-- Name: Topic_pkey; Type: CONSTRAINT; Schema: public; Owner: naeemul
--

ALTER TABLE ONLY topic
    ADD CONSTRAINT "Topic_pkey" PRIMARY KEY (id);


--
-- Name: sentence_pkey; Type: CONSTRAINT; Schema: public; Owner: naeemul
--

ALTER TABLE ONLY sentence
    ADD CONSTRAINT sentence_pkey PRIMARY KEY (id);


--
-- Name: public; Type: ACL; Schema: -; Owner: postgres
--

REVOKE ALL ON SCHEMA public FROM PUBLIC;
REVOKE ALL ON SCHEMA public FROM postgres;
GRANT ALL ON SCHEMA public TO postgres;
GRANT ALL ON SCHEMA public TO PUBLIC;


--
-- PostgreSQL database dump complete
--

