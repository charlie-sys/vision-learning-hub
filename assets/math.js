/* Render authored LaTeX using the locally bundled KaTeX assets. */
document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll("[data-tex]").forEach((element) => {
    try {
      katex.render(element.dataset.tex, element, {
        displayMode: element.classList.contains("math-display"),
        throwOnError: true,
        trust: false,
        output: "htmlAndMathml",
      });
      element.dataset.rendered = "true";
    } catch (error) {
      element.classList.add("math-error");
      element.dataset.rendered = "false";
      console.error("Formula rendering failed:", error.message);
    }
  });
});
