?($	?~:5?!???W?ham???T???N??!?c?]K???$	.?E??<@h7??v?@s?6???!??vK?1@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$????B????J?4??A%??C???Y?E???Ԩ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?*??	???.n????A㥛? ???Y?q??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&|??Pk???;pΈ????Ar?鷯??Y??Pk?w??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&S??:????"??~??An????Y??&???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??C?l???[Ӽ???Avq?-??Y?ݓ??Z??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
ףp=
???0?*???A;M?O??Y??H?}??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?HP?????m4????A?c]?F??YHP?s??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&|a2U0*??~??k	???A??:M??YL7?A`???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?}8gD?????z6??A)\???(??Y?j+??ݓ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	????z??}?5^?I??A?$??C??Yw-!?l??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
&S????RI??&???A???z6??Ya??+e??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?c?]K???????ׁ??A??n????Y??ZӼ???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&M?O????\?C????A???????Y?b?=y??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?I+??? ?o_???AB?f??j??Y??z6???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?? ?rh??o?ŏ1??A??(???YJ+???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&ףp=
???|??Pk???A#J{?/L??YV-???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?=?U???+????A2??%????Y%u???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?4?8EG???-???1??A??	h"l??Y??ǘ????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?&1???`??"????A?k	??g??Y??H?}??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
h"lxz??r?鷯??A??A?f??Y?<,Ԛ???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?T???N????_vO??A?H?}??Y??_?L??*	43333??@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatV????_??!???? B@)???N@??1<~???@@:Preprocessing2F
Iterator::Model?s?????!g9???jB@)(??y??1?"???6@:Preprocessing2U
Iterator::Model::ParallelMapV2[????<??!??Κ7?+@)[????<??1??Κ7?+@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??_vO??!??IQz?O@)@?߾???1O8b6#@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSliceڬ?\mŮ?!?Y)?%@)ڬ?\mŮ?1?Y)?%@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate	?c???!??#??b*@),e?X??1?-n??@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?Q?????!?fo???1@)?5?;Nѡ?1kv??t@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*Dio??ɔ?!(m???]@)Dio??ɔ?1(m???]@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 52.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9Om?hN?@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	????3??w?=??????_vO??![Ӽ???	!       "	!       *	!       2$	??b??d??j?+Ε???H?}??!??n????:	!       B	!       J$	??~???????H?????Pk?w??!??z6???R	!       Z$	??~???????H?????Pk?w??!??z6???JCPU_ONLYYOm?hN?@b Y      Y@q?M??@@"?
both?Your program is POTENTIALLY input-bound because 52.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?32.223% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 