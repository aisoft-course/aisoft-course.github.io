(async function () {
  try {
    const res = await fetch('https://api.github.com/repos/aisoft-course/aisoft-course.github.io')
    if (res.ok) {
      const data = await res.json()
      const count = data.stargazers_count
      if (count !== undefined) {
        const el = document.getElementById('github-star-count')
        if (el) el.textContent = count
      }
    }
  } catch (e) {}
})()
