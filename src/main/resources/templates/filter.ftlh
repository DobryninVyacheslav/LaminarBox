<#import "parts/common.ftlh" as c>

<@c.page>
    <h4>Предсказание загрязнения фильтров
        <hr/>
    </h4>

    <div class="form-check form-switch mt-5 mb-5">
        <input class="form-check-input" type="checkbox" id="glass" checked>
        <label class="form-check-label" for="glass">Открыть стекло</label>
    </div>
    <div class="form-check form-switch mt-5 mb-5">
        <input class="form-check-input" type="checkbox" id="air" checked>
        <label class="form-check-label" for="air">Открыть заслонку на блоке вентилятора</label>
    </div>

    <div class="input-group mt-5 mb-3 w-50">
        <input type="text" class="form-control" placeholder="Давление (от 200 до 600 Па)" aria-label="Username"
               aria-describedby="basic-addon1" aria-valuemin="200" aria-valuemax="600" id="pressure">
        <span class="input-group-text">Давление, создаваемое вентилятором</span>
    </div>

    <button type="button" class="btn btn-primary mt-5 mb-5" id="get-time">Получить время</button>


    <div id="result">Нажмите кнопку</div>

    <script src="/filter.js"></script>
</@c.page>