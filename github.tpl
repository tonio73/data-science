{% extends "full.tpl" %}

{% block title %}Notebooks and Python about data science{% endblock %}

{% block header %}
    {{ super() }}
    <script async defer src="https://buttons.github.io/buttons.js"></script>
    <link rel="stylesheet" type="text/css" media="screen" href="/data-science/assets/css/style.css?v=9033ce03a74e40fe7a9f1cd464f761ff7c5f4711">
    <style>
        #project_star {
            color: #fff;
            font-size: 16px;
            font-weight: 300;
            background: none;
         }
    </style>
{% endblock %}

{% block body %}
    <div id="header_wrap" class="outer">
        <header class="inner">
          <a id="forkme_banner" href="https://github.com/tonio73/data-science">View on GitHub</a>

          <h1 id="project_title"><a href="/data-science">data-science</a></h1>
          <h2 id="project_tagline">Notebooks and Python about data science</h2>
          <h5 id="project_star">If you like this project please add your <a class="github-button" href="https://github.com/tonio73/data-science" data-icon="octicon-star" aria-label="Star tonio73/data-science on GitHub">Star</a>
          </h5>
        </header>
    </div>
    
    <div id="main_content_wrap" class="outer">
      <section id="main_content">
        {{ super() }}
      </section>
    </div>
    
    <div id="footer_wrap" class="outer">
      <footer class="inner">
        <h5 id="project_star">If you like this project please add your <a class="github-button" href="https://github.com/tonio73/data-science" data-icon="octicon-star" aria-label="Star tonio73/data-science on GitHub">Star</a>
          </h5>
        <p class="copyright">data-science maintained by <a href="https://github.com/tonio73">tonio73</a></p>
      </footer>
    </div>
{% endblock %}