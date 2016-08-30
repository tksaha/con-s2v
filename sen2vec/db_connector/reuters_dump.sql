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
    text text,
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
    text text,
    file text,
    metadata text
);


ALTER TABLE document OWNER TO naeemul;

--
-- Name: document_topic; Type: TABLE; Schema: public; Owner: naeemul
--

CREATE TABLE document_topic (
    document_id integer,
    topic_id integer
);


ALTER TABLE document_topic OWNER TO naeemul;

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
-- Name: public; Type: ACL; Schema: -; Owner: postgres
--

REVOKE ALL ON SCHEMA public FROM PUBLIC;
REVOKE ALL ON SCHEMA public FROM postgres;
GRANT ALL ON SCHEMA public TO postgres;
GRANT ALL ON SCHEMA public TO PUBLIC;


--
-- PostgreSQL database dump complete
--

