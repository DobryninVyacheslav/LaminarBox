<#import "parts/common.ftlh" as c>

<@c.page>
    <div class="bg-light p-5 rounded">
        <h1>Контроль технологического процесса</h1>
        <p class="fs-5" id="status"></p>

        <div class="row mt-4">
            <div class="col-sm">
                <div class="controls">
                    <div class="row">
                        <div class="col-md-4">
                            <button class="btn btn-primary" id="button">Выбрать камеру</button>
                        </div>
                        <div class="col-sm">
                            <select class="form-select" id="select">
                                <option></option>
                            </select>
                        </div>
                    </div>
                </div>
            </div>
            <div class="col-sm">
                <video autoplay playsinline muted id="webcam" width="180" height="180"
                       class="position-relative top-50 start-50 translate-middle"></video>
            </div>
        </div>

    </div>


    <script src="/process.js" charset="utf-8"></script>
</@c.page>

