?&$	֭G?q'????G`?Ӡ??N?b??!??]ؚ??$	Wk???
@q$?`?????S?@!??
??@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails02s??cM?????-I??AO?P??&??Y?r??h???rtrain 81"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0"?T3k)??f??C???A??u????Ya?N"¿??rtrain 82"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails03?`??i??x?'-\V??Am????Y??n????rtrain 83"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?M֨???fd????Avp?71$??Yh?.?K??rtrain 84"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0p?????S??????A?*?]gC??YF?=?Ӟ??rtrain 85"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0??]ؚ??1Xr???A:τ&???Y???LL??rtrain 86"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?6?Nx??????cZ???AU????,??Y?? ?S???rtrain 87"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0	?`??
???6x_????A??a????Y???????rtrain 88"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0
???5????}?ݮ????A??;???Y?E}?;l??rtrain 89"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails03??L??&R???0??AU?2?F??Y??M?q??rtrain 90"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0`<???	????j?????A???????Y?0~????rtrain 91"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?3?ތ???RG??Ȯ??Ad?? w??Y?1k?M??rtrain 92"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0;??.R(??6??????A?^ ??Y*Ŏơ~??rtrain 93"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?9]???9_??????A!V?a???Y3?뤾,??rtrain 94"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?N?b??pa?xwd??A?gA(????Y?6?ח?rtrain 95"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0??W?<??????????Am?/?r??Y?x[??٘?rtrain 96"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0˿?W????W\????A?lW??e??Yܺ??:???rtrain 97*	֣p=
R?@2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat??% ????!R??T?@)??\?????1?????<@:Preprocessing2E
Iterator::Root?eN?????!?c۰pC@)?$xC??1??k???6@:Preprocessing2T
Iterator::Root::ParallelMapV2ԁ??V_??!??[&?0@)ԁ??V_??1??[&?0@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip?v??/??!? ?$O?N@)w????@??1?_?ݾ#@:Preprocessing2?
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSliceK?8??լ?!N?%?8"@)K?8??լ?1N?%?8"@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap??,???!????/4@)np?????1*??6?g@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[2]::ConcatenateV}??b??!?u?YH*+@)ĔH??Q??1Ĭ[m?@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*???\7???!M?'C@)???\7???1M?'C@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 43.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9??8?
@I??9??(X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	??C'?b??!1?IW??&R???0??!??j?????	!       "	!       *	!       2$	???N???E9??u????gA(????!:τ&???:	!       B	!       J$	?<?(????񡫥j????????!3?뤾,??R	!       Z$	?<?(????񡫥j????????!3?뤾,??b	!       JCPU_ONLYY??8?
@b q??9??(X@Y      Y@q}ٞyR@"?	
both?Your program is POTENTIALLY input-bound because 43.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?72.4% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 